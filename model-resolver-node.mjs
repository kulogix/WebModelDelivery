#!/usr/bin/env node
/**
 * model-resolver-node.mjs — Node.js equivalent of model-sw.js
 *
 * Transparently intercepts model file loading in Node.js by monkey-patching
 * globalThis.fetch. Libraries like @huggingface/transformers (Node.js) and
 * node-llama-cpp that use fetch() internally will have their requests
 * intercepted, resolved via filemap.json, and served from reassembled shards
 * — exactly like the browser Service Worker does.
 *
 * Supports BOTH remote CDN sources and local flat-repo directories.
 *
 * Also provides a resolve() API for libraries that load from file paths
 * (e.g., llama-cpp-python's Node bindings, onnxruntime-node).
 *
 * ─── Usage: Fetch Interception (Transformers.js Node) ──────────────────────
 *
 *   import { ModelResolver } from './model-resolver-node.mjs';
 *
 *   const resolver = new ModelResolver({ cacheDir: './.model-cache' });
 *
 *   // Remote CDN source:
 *   resolver.addSource({
 *     pathPrefix: '/models/embedding/',
 *     cdnBase: 'https://cdn.jsdelivr.net/gh/user/cdn-embedding@v1',
 *     manifest: 'q4f16',
 *     progress: true,
 *   });
 *
 *   // OR local flat-repo source (no network needed):
 *   resolver.addSource({
 *     pathPrefix: '/models/embedding/',
 *     localBase: '/path/to/pkg-embedding',   // directory with filemap.json
 *     manifest: 'q4f16',
 *     progress: true,
 *   });
 *
 *   resolver.installFetchHook();
 *
 *   // Now Transformers.js works exactly like the browser version:
 *   import { AutoModel, AutoTokenizer } from '@huggingface/transformers';
 *   const tokenizer = await AutoTokenizer.from_pretrained('/models/embedding/');
 *   const model = await AutoModel.from_pretrained('/models/embedding/', {
 *     dtype: 'q4f16',
 *   });
 *
 * ─── Usage: Direct Resolve (file-path consumers) ──────────────────────────
 *
 *   // From CDN:
 *   const localDir = await resolver.resolve(
 *     'https://cdn.jsdelivr.net/gh/user/cdn-embedding@v1',
 *     { manifest: 'q4f16' }
 *   );
 *
 *   // From local flat repo (reassembles shards to cache dir):
 *   const localDir = await resolver.resolve(
 *     '/path/to/pkg-embedding',
 *     { manifest: 'q4f16' }
 *   );
 *
 * ─── Usage: GGUF Resolve (llama.cpp bindings) ─────────────────────────────
 *
 *   const files = await resolver.resolveFiles(
 *     '/path/to/pkg-llm',
 *     { manifest: 'q4_0' }
 *   );
 *   // files = { 'model-q4_0.gguf': '/abs/path/to/cache/model-q4_0.gguf', ... }
 */

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

// ─── Helpers ────────────────────────────────────────────────────────────────

function isLocalPath(str) {
  if (!str) return false;
  if (/^(\/|\.\/|\.\.\/)/.test(str)) return true;
  if (/^[A-Za-z]:[\\/]/.test(str)) return true;
  if (str.startsWith('file://')) return true;
  if (/^https?:\/\//.test(str)) return false;
  return true;
}

function toLocalPath(str) {
  if (str.startsWith('file://')) return new URL(str).pathname;
  return str;
}

// ─── ModelResolver ─────────────────────────────────────────────────────────

export class ModelResolver {
  /**
   * @param {Object} opts
   * @param {string} [opts.cacheDir='./.model-cache'] — local cache directory
   * @param {boolean} [opts.verifySha256=false] — verify SHA256 after reassembly
   * @param {number} [opts.retries=3] — retry count per shard download
   */
  constructor(opts = {}) {
    this.cacheDir = path.resolve(opts.cacheDir || './.model-cache');
    this.verifySha256 = opts.verifySha256 ?? false;
    this.retries = opts.retries ?? 3;

    this._sources = [];
    this._filemaps = new Map();
    this._filemapLoading = new Map();
    this._originalFetch = null;
    this._hooked = false;
    this._progressState = new Map();
    this._onProgress = null;
  }

  // ─── Source Configuration ───────────────────────────────────────────────

  /**
   * Add a source for fetch interception.
   * Provide EITHER cdnBase (remote) or localBase (local filesystem), not both.
   */
  addSource(source) {
    if (!source.cdnBase && !source.localBase) {
      throw new Error('addSource requires either cdnBase or localBase');
    }

    const s = {
      pathPrefix: source.pathPrefix.endsWith('/') ? source.pathPrefix : source.pathPrefix + '/',
      cdnBase: source.cdnBase ? source.cdnBase.replace(/\/+$/, '') : '',
      localBase: source.localBase ? path.resolve(toLocalPath(source.localBase)) : '',
      manifest: source.manifest || '',
      progress: !!source.progress,
      onProgress: source.onProgress || null,
    };
    s._sourceKey = s.localBase || s.cdnBase;

    this._sources.push(s);

    if (s.progress) {
      this._progressState.set(s.pathPrefix, {
        mode: s.manifest ? 'explicit' : 'adaptive',
        totalBytes: 0, loadedBytes: 0,
        files: new Map(), activeFiles: new Set(),
        candidates: [], selectedManifest: s.manifest || null,
        finalized: false, lastFile: '',
      });
    }

    this._loadFilemap(s._sourceKey, !!s.localBase);
  }

  set onProgress(fn) { this._onProgress = fn; }

  // ─── Fetch Hook ────────────────────────────────────────────────────────

  installFetchHook() {
    if (this._hooked) return;
    this._originalFetch = globalThis.fetch;
    const self = this;

    globalThis.fetch = async function(input, init) {
      const url = typeof input === 'string' ? input
        : input instanceof URL ? input.href
        : input?.url || '';

      const match = self._matchSource(url);
      if (match) return self._handleFetch(url, match.source, match.relPath, init);
      return self._originalFetch.call(globalThis, input, init);
    };
    this._hooked = true;
  }

  removeFetchHook() {
    if (!this._hooked) return;
    globalThis.fetch = this._originalFetch;
    this._hooked = false;
  }

  // ─── Direct Resolve API ────────────────────────────────────────────────

  /**
   * Download/reassemble all files from a source into a local directory.
   * @param {string} source — CDN URL root OR local flat-repo directory path
   */
  async resolve(source, opts = {}) {
    const local = isLocalPath(source);
    const sourceKey = local ? path.resolve(toLocalPath(source)) : source.replace(/\/+$/, '');
    const filemap = await this._loadFilemap(sourceKey, local);
    if (!filemap) throw new Error(`Failed to load filemap from ${source}`);

    const fileList = this._getFileList(filemap, opts.manifest);
    const outDir = this._cachePathForSource(sourceKey, opts.manifest);
    fs.mkdirSync(outDir, { recursive: true });

    const totalBytes = fileList.reduce((sum, vp) => sum + (filemap.files[vp]?.size || 0), 0);
    let loadedBytes = 0;

    for (const vp of fileList) {
      const entry = filemap.files[vp];
      if (!entry) continue;

      const outPath = path.join(outDir, vp);
      fs.mkdirSync(path.dirname(outPath), { recursive: true });

      if (fs.existsSync(outPath) && fs.statSync(outPath).size === entry.size) {
        loadedBytes += entry.size;
        opts.onProgress?.({ percent: Math.round(loadedBytes / totalBytes * 100),
          loaded: loadedBytes, total: totalBytes, file: vp, done: false });
        continue;
      }

      await this._reassembleFile(sourceKey, local, vp, entry, outPath, (bytes) => {
        loadedBytes += bytes;
        opts.onProgress?.({ percent: Math.round(loadedBytes / totalBytes * 100),
          loaded: loadedBytes, total: totalBytes, file: vp, done: false });
      });
    }

    opts.onProgress?.({ percent: 100, loaded: totalBytes, total: totalBytes, file: '', done: true });
    return outDir;
  }

  async resolveFiles(source, opts = {}) {
    const dir = await this.resolve(source, opts);
    const sourceKey = isLocalPath(source) ? path.resolve(toLocalPath(source)) : source.replace(/\/+$/, '');
    const filemap = this._filemaps.get(sourceKey);
    const fileList = this._getFileList(filemap, opts.manifest);
    const result = {};
    for (const vp of fileList) result[vp] = path.join(dir, vp);
    return result;
  }

  // ─── Internal: Source Matching ─────────────────────────────────────────

  _matchSource(url) {
    let pathname;
    try { pathname = new URL(url, 'http://localhost').pathname; } catch { pathname = url; }

    for (const s of this._sources) {
      if (pathname.startsWith(s.pathPrefix)) {
        const relPath = pathname.slice(s.pathPrefix.length);
        if (relPath) return { source: s, relPath };
      }
    }
    return null;
  }

  // ─── Internal: Fetch Handling ──────────────────────────────────────────

  async _handleFetch(url, source, relPath, init) {
    const filemap = await this._loadFilemap(source._sourceKey, !!source.localBase);
    if (!filemap) {
      if (source.localBase) return new Response(null, { status: 404 });
      return this._originalFetch(`${source.cdnBase}/${relPath}`, init);
    }

    const entry = filemap.files[relPath];
    if (!entry) {
      if (source.localBase) {
        const fp = path.join(source.localBase, relPath);
        if (fs.existsSync(fp)) {
          const buf = fs.readFileSync(fp);
          return new Response(buf, { status: 200, headers: {
            'Content-Type': 'application/octet-stream', 'Content-Length': String(buf.length) } });
        }
        return new Response(null, { status: 404 });
      }
      return this._originalFetch(`${source.cdnBase}/${relPath}`, init);
    }

    if (source.progress) this._trackFile(source.pathPrefix, relPath, entry.size);

    // Unsharded
    if (!entry.shards) {
      const cdnFile = entry.cdn_file || relPath;
      let resp;
      if (source.localBase) {
        const buf = this._readLocal(source.localBase, cdnFile);
        resp = new Response(buf, { status: 200, headers: {
          'Content-Type': 'application/octet-stream', 'Content-Length': String(buf.length) } });
      } else {
        resp = await this._fetchWithCache(`${source.cdnBase}/${cdnFile}`);
      }
      this._recordProgress(source, relPath, entry.size);
      return resp;
    }

    // Sharded — check Range header
    const rh = init?.headers?.get?.('Range')
      || init?.headers?.Range
      || (typeof init?.headers === 'object' && !Array.isArray(init?.headers) ? init?.headers?.range : null);

    if (rh) {
      const m = rh.match(/bytes=(\d+)-(\d*)/);
      if (m) {
        return this._respondRange(entry, source, parseInt(m[1],10),
          m[2] !== '' ? parseInt(m[2],10) : entry.size - 1, relPath);
      }
    }
    return this._respondFull(entry, source, relPath);
  }

  async _respondFull(entry, source, relPath) {
    const buffers = [];
    for (const shard of entry.shards) {
      const buf = source.localBase
        ? this._readLocal(source.localBase, shard.file)
        : await this._fetchShardBuf(`${source.cdnBase}/${shard.file}`);
      buffers.push(buf);
      this._recordProgress(source, relPath, shard.size);
    }
    const combined = Buffer.concat(buffers);
    return new Response(combined, { status: 200, headers: {
      'Content-Type': 'application/octet-stream', 'Content-Length': String(entry.size),
      'Accept-Ranges': 'bytes', 'X-Model-Resolver': 'reassembled' } });
  }

  async _respondRange(entry, source, start, end, relPath) {
    end = Math.min(end, entry.size - 1);
    if (start > end || start >= entry.size)
      return new Response(null, { status: 416, headers: { 'Content-Range': `bytes */${entry.size}` } });

    const length = end - start + 1;
    const parts = [];
    for (const shard of entry.shards) {
      const sEnd = shard.offset + shard.size - 1;
      if (sEnd < start) continue;
      if (shard.offset > end) break;
      parts.push({ file: shard.file, subStart: Math.max(start, shard.offset) - shard.offset,
        subEnd: Math.min(end, sEnd) - shard.offset, shardSize: shard.size });
    }
    if (!parts.length)
      return new Response(null, { status: 416, headers: { 'Content-Range': `bytes */${entry.size}` } });

    const buffers = [];
    for (const p of parts) {
      const full = source.localBase
        ? this._readLocal(source.localBase, p.file)
        : await this._fetchShardBuf(`${source.cdnBase}/${p.file}`);
      buffers.push(p.subStart === 0 && p.subEnd === p.shardSize - 1
        ? full : full.subarray(p.subStart, p.subEnd + 1));
    }
    this._recordProgress(source, relPath, length);

    return new Response(Buffer.concat(buffers), { status: 206, headers: {
      'Content-Type': 'application/octet-stream',
      'Content-Range': `bytes ${start}-${end}/${entry.size}`,
      'Content-Length': String(length), 'Accept-Ranges': 'bytes',
      'X-Model-Resolver': 'range' } });
  }

  // ─── Internal: File I/O ───────────────────────────────────────────────

  _readLocal(base, filename) {
    const fp = path.join(base, filename);
    if (!fs.existsSync(fp)) throw new Error(`[model-resolver] Local file not found: ${fp}`);
    return fs.readFileSync(fp);
  }

  async _fetchShardBuf(url) {
    const cp = this._shardCachePath(url);
    if (fs.existsSync(cp)) return fs.readFileSync(cp);

    let lastErr;
    const fetchFn = this._originalFetch || globalThis.fetch;
    for (let i = 0; i < this.retries; i++) {
      try {
        const r = await fetchFn(url);
        if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
        const buf = Buffer.from(await r.arrayBuffer());
        fs.mkdirSync(path.dirname(cp), { recursive: true });
        fs.writeFileSync(cp, buf);
        return buf;
      } catch (e) {
        lastErr = e;
        if (i < this.retries - 1) await new Promise(r => setTimeout(r, 1000 * (i + 1)));
      }
    }
    throw lastErr;
  }

  async _fetchWithCache(url) {
    const buf = await this._fetchShardBuf(url);
    return new Response(buf, { status: 200, headers: {
      'Content-Type': 'application/octet-stream', 'Content-Length': String(buf.length) } });
  }

  _shardCachePath(url) {
    return path.join(this.cacheDir, 'shards',
      crypto.createHash('sha256').update(url).digest('hex').slice(0, 16) + '_' + url.split('/').pop());
  }

  _cachePathForSource(key, manifest) {
    const h = crypto.createHash('sha256').update(key).digest('hex').slice(0, 12);
    return path.join(this.cacheDir, 'resolved', `${h}${manifest ? '_' + manifest : ''}`);
  }

  // ─── Internal: File Reassembly ────────────────────────────────────────

  async _reassembleFile(sourceKey, isLocal, vp, entry, outPath, onBytes) {
    if (!entry.shards) {
      const cdnFile = entry.cdn_file || vp;
      const buf = isLocal ? this._readLocal(sourceKey, cdnFile)
        : await this._fetchShardBuf(`${sourceKey}/${cdnFile}`);
      fs.writeFileSync(outPath, buf);
      onBytes?.(entry.size);
      return;
    }

    const fd = fs.openSync(outPath, 'w');
    try {
      for (const shard of entry.shards) {
        const buf = isLocal ? this._readLocal(sourceKey, shard.file)
          : await this._fetchShardBuf(`${sourceKey}/${shard.file}`);
        fs.writeSync(fd, buf, 0, buf.length, shard.offset);
        onBytes?.(shard.size);
      }
    } finally { fs.closeSync(fd); }

    if (this.verifySha256 && entry.sha256) {
      const hash = crypto.createHash('sha256');
      const stream = fs.createReadStream(outPath);
      for await (const chunk of stream) hash.update(chunk);
      const actual = hash.digest('hex');
      if (actual !== entry.sha256) {
        fs.unlinkSync(outPath);
        throw new Error(`SHA256 mismatch for ${vp}: expected ${entry.sha256}, got ${actual}`);
      }
    }
  }

  // ─── Internal: Filemap Loading ────────────────────────────────────────

  async _loadFilemap(sourceKey, isLocal = false) {
    if (this._filemaps.has(sourceKey)) return this._filemaps.get(sourceKey);

    if (!this._filemapLoading.has(sourceKey)) {
      this._filemapLoading.set(sourceKey, (async () => {
        try {
          let data;
          if (isLocal) {
            const fp = path.join(sourceKey, 'filemap.json');
            if (!fs.existsSync(fp)) throw new Error(`filemap.json not found in ${sourceKey}`);
            data = JSON.parse(fs.readFileSync(fp, 'utf8'));
          } else {
            const cp = path.join(this.cacheDir, 'filemaps',
              crypto.createHash('sha256').update(sourceKey).digest('hex').slice(0, 16) + '.json');
            if (fs.existsSync(cp)) {
              data = JSON.parse(fs.readFileSync(cp, 'utf8'));
            } else {
              const fetchFn = this._originalFetch || globalThis.fetch;
              const r = await fetchFn(`${sourceKey}/filemap.json`);
              if (!r.ok) throw new Error(`HTTP ${r.status}`);
              data = await r.json();
              fs.mkdirSync(path.dirname(cp), { recursive: true });
              fs.writeFileSync(cp, JSON.stringify(data, null, 2));
            }
          }
          this._filemaps.set(sourceKey, data);
          this._initProgressFromFilemap(sourceKey, data);
          return data;
        } catch (err) {
          console.error(`[model-resolver] Filemap failed for ${sourceKey}:`, err.message);
          this._filemapLoading.delete(sourceKey);
          return null;
        }
      })());
    }
    return this._filemapLoading.get(sourceKey);
  }

  _getFileList(filemap, manifestName) {
    if (manifestName && filemap.manifests?.[manifestName])
      return filemap.manifests[manifestName].files;
    return Object.keys(filemap.files);
  }

  // ─── Progress Tracking ────────────────────────────────────────────────

  _initProgressFromFilemap(sourceKey, filemap) {
    const source = this._sources.find(s => s._sourceKey === sourceKey);
    if (!source?.progress) return;
    const st = this._progressState.get(source.pathPrefix);
    if (!st) return;

    const manifests = filemap.manifests || {};
    const names = Object.keys(manifests);

    if (st.mode === 'explicit') {
      const m = manifests[st.selectedManifest];
      this._setFiles(st, filemap, m ? m.files : Object.keys(filemap.files));
    } else if (names.length > 0) {
      st.mode = 'adaptive';
      st.candidates = [...names];
      const largest = names.reduce((a, b) => manifests[a].size >= manifests[b].size ? a : b);
      st.selectedManifest = largest;
      this._setFiles(st, filemap, manifests[largest].files);
    } else {
      st.mode = 'fallback';
      this._setFiles(st, filemap, Object.keys(filemap.files));
    }
  }

  _setFiles(st, filemap, vpList) {
    st.totalBytes = 0; st.files.clear();
    for (const vp of vpList) {
      const e = filemap.files[vp];
      if (e) { st.files.set(vp, { size: e.size, loaded: 0 }); st.totalBytes += e.size; }
    }
  }

  _trackFile(pp, rp, sz) {
    const st = this._progressState.get(pp);
    if (!st) return;
    st.activeFiles.add(rp);
    if (!st.files.has(rp)) { st.files.set(rp, { size: sz, loaded: 0 }); st.totalBytes += sz; }
  }

  _recordProgress(source, relPath, bytes) {
    if (!source.progress) return;
    const st = this._progressState.get(source.pathPrefix);
    if (!st) return;
    const f = st.files.get(relPath);
    if (!f) return;
    const prev = f.loaded;
    f.loaded = Math.min(f.loaded + bytes, f.size);
    const delta = f.loaded - prev;
    if (delta > 0) { st.loadedBytes += delta; st.lastFile = relPath; }

    const msg = {
      pathPrefix: source.pathPrefix, file: relPath,
      fileLoaded: f.loaded, fileTotal: f.size,
      modelLoaded: st.loadedBytes, modelTotal: st.totalBytes,
      percent: st.totalBytes > 0 ? Math.min(100, Math.round(st.loadedBytes / st.totalBytes * 100)) : 0,
      done: st.finalized || (st.loadedBytes >= st.totalBytes && st.totalBytes > 0),
      manifest: st.selectedManifest, mode: st.mode,
    };
    source.onProgress?.(msg);
    this._onProgress?.(msg);
  }

  // ─── Utility ─────────────────────────────────────────────────────────

  clearCache() {
    if (fs.existsSync(this.cacheDir)) fs.rmSync(this.cacheDir, { recursive: true });
    this._filemaps.clear(); this._filemapLoading.clear();
  }

  getCacheStats() {
    const countDir = (dir) => {
      let count = 0, bytes = 0;
      if (fs.existsSync(dir)) {
        const walk = (d) => { for (const f of fs.readdirSync(d)) {
          const fp = path.join(d, f), stat = fs.statSync(fp);
          if (stat.isDirectory()) walk(fp); else { count++; bytes += stat.size; }
        }};
        walk(dir);
      }
      return { count, bytes };
    };
    const shards = countDir(path.join(this.cacheDir, 'shards'));
    const resolved = countDir(path.join(this.cacheDir, 'resolved'));
    return { shards: shards.count, shardBytes: shards.bytes,
      resolvedFiles: resolved.count, resolvedBytes: resolved.bytes,
      totalBytes: shards.bytes + resolved.bytes };
  }
}

// ─── Convenience ────────────────────────────────────────────────────────────

export async function resolveModel(source, opts = {}) {
  const resolver = new ModelResolver({ cacheDir: opts.cacheDir || './.model-cache', verifySha256: opts.verifySha256 });
  return resolver.resolve(source, opts);
}

export default ModelResolver;
