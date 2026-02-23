/**
 * model-sw.js v4 — Service Worker for transparent CDN model delivery
 *
 * v4 changes:
 *   - Adaptive manifest detection: starts at worst-case (largest manifest),
 *     narrows as model-specific files are requested, idle-finalizes.
 *   - Explicit manifest mode: page says manifest:'q4f16', denominator fixed.
 *   - Fallback: no manifests in filemap → tracks all files.
 *   - Progress throttled ~4/sec, clamped per-file, zero overhead if disabled.
 *   - MODEL_SW_COMPLETE: page can force 100%/done at any time.
 *
 * ─── Progress messages ──────────────────────────────────────────────────────
 *
 *   {
 *     type: 'MODEL_SW_PROGRESS',
 *     pathPrefix, file, fileLoaded, fileTotal,
 *     modelLoaded, modelTotal, percent, done,
 *     manifest,  // currently selected manifest name (or null)
 *     mode,      // 'explicit' | 'adaptive' | 'fallback'
 *   }
 */

// ─── State ──────────────────────────────────────────────────────────────────

let sources = [];
const filemaps = new Map();
const filemapLoading = new Map();
const shardInflight = new Map();   // url → Promise<ArrayBuffer> — dedup concurrent fetches
const CACHE_NAME = 'model-shards-v1';

// ─── Progress ───────────────────────────────────────────────────────────────

const progressState = new Map();
const PROGRESS_THROTTLE_MS = 250;
const IDLE_TIMEOUT_MS = 2000;

// ─── Lifecycle ──────────────────────────────────────────────────────────────

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

// ─── Messages ───────────────────────────────────────────────────────────────

self.addEventListener('message', (event) => {
  const { data } = event;

  if (data?.type === 'MODEL_SW_INIT') {
    sources = (data.sources || []).map(s => ({
      pathPrefix: s.pathPrefix.endsWith('/') ? s.pathPrefix : s.pathPrefix + '/',
      cdnBase: s.cdnBase.replace(/\/+$/, ''),
      progress: !!s.progress,
      manifest: s.manifest || '',
    }));

    filemaps.clear();
    filemapLoading.clear();
    for (const [, st] of progressState) {
      if (st.broadcastTimer != null) clearTimeout(st.broadcastTimer);
      if (st.idleTimer != null) clearTimeout(st.idleTimer);
    }
    progressState.clear();

    for (const s of sources) {
      loadFilemap(s.cdnBase);
      if (s.progress) {
        progressState.set(s.pathPrefix, makeProgressState(s.manifest));
      }
    }
    console.log('[model-sw] v4 init:', sources.map(s =>
      s.pathPrefix + (s.progress ? ` (${s.manifest || 'adaptive'})` : '')
    ));
  }

  if (data?.type === 'MODEL_SW_CLEAR_CACHE') {
    caches.delete(CACHE_NAME).then(() => {
      event.source?.postMessage({ type: 'MODEL_SW_CACHE_CLEARED' });
    });
  }

  if (data?.type === 'MODEL_SW_STATUS') {
    event.source?.postMessage({
      type: 'MODEL_SW_STATUS',
      sources: sources.map(s => s.pathPrefix),
      filemapsLoaded: [...filemaps.keys()],
    });
  }

  if (data?.type === 'MODEL_SW_COMPLETE') {
    const st = progressState.get(data.pathPrefix);
    if (st) finalizeProgress(data.pathPrefix, st);
  }
});

function makeProgressState(manifestName) {
  return {
    mode: manifestName ? 'explicit' : 'adaptive',
    totalBytes: 0, loadedBytes: 0,
    files: new Map(),
    activeFiles: new Set(),
    candidates: [],
    selectedManifest: manifestName || null,
    pendingFetches: 0,
    idleTimer: null,
    finalized: false,
    lastBroadcast: 0,
    broadcastTimer: null,
    lastFile: '',
  };
}

// ─── Filemap → progress init ────────────────────────────────────────────────

function initProgressFromFilemap(cdnBase, filemap) {
  const source = sources.find(s => s.cdnBase === cdnBase);
  if (!source?.progress) return;
  const st = progressState.get(source.pathPrefix);
  if (!st) return;

  const manifests = filemap.manifests || {};
  const names = Object.keys(manifests);

  if (st.mode === 'explicit') {
    const m = manifests[st.selectedManifest];
    if (m) {
      setFilesFromList(st, filemap, m.files);
      console.log(`[model-sw] Explicit: "${st.selectedManifest}" ${(st.totalBytes / 1048576).toFixed(1)} MB`);
    } else {
      console.warn(`[model-sw] Manifest "${st.selectedManifest}" not found → all files`);
      setFilesFromList(st, filemap, Object.keys(filemap.files));
    }
  } else if (names.length > 0) {
    st.mode = 'adaptive';
    st.candidates = [...names];
    const largest = names.reduce((a, b) => manifests[a].size >= manifests[b].size ? a : b);
    st.selectedManifest = largest;
    setFilesFromList(st, filemap, manifests[largest].files);
    console.log(`[model-sw] Adaptive: start "${largest}" (${(st.totalBytes / 1048576).toFixed(1)} MB), ${names.length} candidate(s)`);
  } else {
    st.mode = 'fallback';
    setFilesFromList(st, filemap, Object.keys(filemap.files));
    console.log(`[model-sw] Fallback: all ${st.files.size} files, ${(st.totalBytes / 1048576).toFixed(1)} MB`);
  }
}

function setFilesFromList(st, filemap, vpList) {
  st.totalBytes = 0;
  st.files.clear();
  for (const vp of vpList) {
    const e = filemap.files[vp];
    if (e) { st.files.set(vp, { size: e.size, loaded: 0 }); st.totalBytes += e.size; }
  }
}

// ─── Adaptive narrowing ─────────────────────────────────────────────────────

function narrowManifest(pathPrefix, relPath) {
  const st = progressState.get(pathPrefix);
  if (!st || st.mode !== 'adaptive' || st.finalized) return;
  const source = sources.find(s => s.pathPrefix === pathPrefix);
  if (!source) return;
  const fm = filemaps.get(source.cdnBase);
  if (!fm?.manifests) return;

  const remaining = st.candidates.filter(n => fm.manifests[n]?.files?.includes(relPath));
  if (remaining.length === 0 || remaining.length === st.candidates.length) return;

  st.candidates = remaining;
  const largest = remaining.reduce((a, b) =>
    fm.manifests[a].size >= fm.manifests[b].size ? a : b);

  if (largest !== st.selectedManifest) {
    const prev = st.selectedManifest;
    st.selectedManifest = largest;
    // Save loaded bytes, repopulate from new manifest, restore
    const saved = new Map();
    for (const [vp, f] of st.files) saved.set(vp, f.loaded);
    setFilesFromList(st, fm, fm.manifests[largest].files);
    st.loadedBytes = 0;
    for (const [vp, f] of st.files) {
      const old = saved.get(vp);
      if (old != null) f.loaded = Math.min(old, f.size);
      st.loadedBytes += f.loaded;
    }
    console.log(`[model-sw] Narrowed: "${prev}" → "${largest}" (${remaining.length} left, ${(st.totalBytes / 1048576).toFixed(1)} MB)`);
  }
}

// ─── Idle detection & finalization ──────────────────────────────────────────

function onFetchStart(pathPrefix) {
  const st = progressState.get(pathPrefix);
  if (!st) return;
  st.pendingFetches++;
  if (st.idleTimer != null) { clearTimeout(st.idleTimer); st.idleTimer = null; }
}

function onFetchEnd(pathPrefix) {
  const st = progressState.get(pathPrefix);
  if (!st) return;
  st.pendingFetches = Math.max(0, st.pendingFetches - 1);
  if (st.pendingFetches === 0 && !st.finalized && st.mode !== 'explicit') {
    if (st.idleTimer != null) clearTimeout(st.idleTimer);
    st.idleTimer = setTimeout(() => {
      st.idleTimer = null;
      if (st.pendingFetches === 0 && !st.finalized) finalizeProgress(pathPrefix, st);
    }, IDLE_TIMEOUT_MS);
  }
}

function finalizeProgress(pathPrefix, st) {
  if (st.finalized) return;
  st.finalized = true;
  if (st.idleTimer != null) { clearTimeout(st.idleTimer); st.idleTimer = null; }
  if (st.broadcastTimer != null) { clearTimeout(st.broadcastTimer); st.broadcastTimer = null; }

  // Shrink denominator to actual requested files
  let actualTotal = 0;
  for (const vp of st.activeFiles) {
    const f = st.files.get(vp);
    if (f) { actualTotal += f.size; f.loaded = f.size; }
  }
  st.totalBytes = actualTotal || st.loadedBytes;
  st.loadedBytes = st.totalBytes;
  doBroadcast(pathPrefix, st);
  console.log(`[model-sw] Finalized: ${pathPrefix} ${(st.totalBytes / 1048576).toFixed(1)} MB actual`);
}

// ─── Core progress helpers ──────────────────────────────────────────────────

function trackFile(pathPrefix, relPath, fileSize) {
  const st = progressState.get(pathPrefix);
  if (!st) return;
  st.activeFiles.add(relPath);
  if (!st.files.has(relPath)) {
    st.files.set(relPath, { size: fileSize, loaded: 0 });
    st.totalBytes += fileSize;
  }
}

function recordProgress(pathPrefix, relPath, bytes) {
  const st = progressState.get(pathPrefix);
  if (!st) return;
  const f = st.files.get(relPath);
  if (!f) return;
  const prev = f.loaded;
  f.loaded = Math.min(f.loaded + bytes, f.size);
  const delta = f.loaded - prev;
  if (delta > 0) { st.loadedBytes += delta; st.lastFile = relPath; }
}

function scheduleProgress(pathPrefix) {
  const st = progressState.get(pathPrefix);
  if (!st) return;
  const isDone = st.finalized || (st.loadedBytes >= st.totalBytes && st.totalBytes > 0);
  if (isDone) {
    if (st.broadcastTimer != null) { clearTimeout(st.broadcastTimer); st.broadcastTimer = null; }
    doBroadcast(pathPrefix, st);
    return;
  }
  const elapsed = performance.now() - st.lastBroadcast;
  if (elapsed >= PROGRESS_THROTTLE_MS) {
    doBroadcast(pathPrefix, st);
  } else if (st.broadcastTimer == null) {
    st.broadcastTimer = setTimeout(() => {
      st.broadcastTimer = null;
      doBroadcast(pathPrefix, st);
    }, PROGRESS_THROTTLE_MS - elapsed);
  }
}

function doBroadcast(pathPrefix, st) {
  st.lastBroadcast = performance.now();
  const f = st.files.get(st.lastFile);
  const msg = {
    type: 'MODEL_SW_PROGRESS',
    pathPrefix,
    file: st.lastFile,
    fileLoaded: f?.loaded || 0,
    fileTotal: f?.size || 0,
    modelLoaded: st.loadedBytes,
    modelTotal: st.totalBytes,
    percent: st.totalBytes > 0 ? Math.min(100, Math.round(st.loadedBytes / st.totalBytes * 100)) : 0,
    done: st.finalized || (st.loadedBytes >= st.totalBytes && st.totalBytes > 0),
    manifest: st.selectedManifest,
    mode: st.mode,
  };
  self.clients.matchAll({ type: 'window' }).then(cs => {
    for (const c of cs) c.postMessage(msg);
  }).catch(() => {});
}

// ─── COEP-safe proxy fetch (wraps cross-origin response in same-origin) ─────

async function proxiedFetch(url) {
  const r = await fetch(url, { mode: 'cors' });
  if (!r.ok) return new Response(null, { status: r.status, statusText: r.statusText });
  const body = await r.arrayBuffer();
  return new Response(body, {
    status: r.status,
    headers: {
      'Content-Type': r.headers.get('Content-Type') || 'application/octet-stream',
      'Content-Length': String(body.byteLength),
      'X-Model-SW': 'proxied-fallback',
    },
  });
}

// ─── Fetch interception ─────────────────────────────────────────────────────

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  const match = matchSource(url.pathname);
  if (!match) return;
  event.respondWith(handleModelFetch(event.request, match.source, match.relPath));
});

function matchSource(pathname) {
  for (const s of sources) {
    if (pathname.startsWith(s.pathPrefix)) {
      const relPath = pathname.slice(s.pathPrefix.length);
      if (relPath) return { source: s, relPath };
    }
  }
  return null;
}

async function handleModelFetch(request, source, relPath) {
  const map = await loadFilemap(source.cdnBase);
  if (!map) return proxiedFetch(`${source.cdnBase}/${relPath}`);
  const entry = map.files[relPath];
  if (!entry) return proxiedFetch(`${source.cdnBase}/${relPath}`);

  if (source.progress) {
    narrowManifest(source.pathPrefix, relPath);
    trackFile(source.pathPrefix, relPath, entry.size);
    onFetchStart(source.pathPrefix);
  }

  // Unsharded — use same dedup/cache pattern as shards
  if (!entry.shards) {
    const cdnFile = entry.cdn_file || relPath;
    const cdnUrl = `${source.cdnBase}/${cdnFile}`;
    try {
      const body = await fetchShard(cdnFile, source);  // reuse dedup-aware fetcher
      recordProgress(source.pathPrefix, relPath, entry.size);
      scheduleProgress(source.pathPrefix);
      return new Response(body, {
        status: 200,
        headers: {
          'Content-Type': 'application/octet-stream',
          'Content-Length': String(body.byteLength),
          'X-Model-SW': 'proxied',
        },
      });
    } finally {
      if (source.progress) onFetchEnd(source.pathPrefix);
    }
  }

  // Sharded
  const rh = request.headers.get('Range');
  if (!rh) return respondFull(entry, source, relPath);
  const m = rh.match(/bytes=(\d+)-(\d*)/);
  if (!m) return respondFull(entry, source, relPath);
  const start = parseInt(m[1], 10);
  const end = m[2] !== '' ? parseInt(m[2], 10) : entry.size - 1;

  try {
    return await respondRange(entry, source, start, end, relPath);
  } finally {
    if (source.progress) onFetchEnd(source.pathPrefix);
  }
}

// ─── Full response (streaming shards) ───────────────────────────────────────

function respondFull(entry, source, relPath) {
  const pp = source.pathPrefix;
  const doProgress = source.progress;
  const stream = new ReadableStream({
    async start(controller) {
      try {
        for (const shard of entry.shards) {
          const buf = await fetchShard(shard.file, source);
          controller.enqueue(new Uint8Array(buf));
          if (doProgress) { recordProgress(pp, relPath, shard.size); scheduleProgress(pp); }
        }
        controller.close();
      } catch (err) { controller.error(err); }
      finally { if (doProgress) onFetchEnd(pp); }
    }
  });
  return new Response(stream, {
    status: 200,
    headers: {
      'Content-Type': 'application/octet-stream',
      'Content-Length': String(entry.size),
      'Accept-Ranges': 'bytes',
      'X-Model-SW': 'reassembled',
    },
  });
}

// ─── Range response ─────────────────────────────────────────────────────────

async function respondRange(entry, source, start, end, relPath) {
  end = Math.min(end, entry.size - 1);
  if (start > end || start >= entry.size)
    return new Response(null, { status: 416, headers: { 'Content-Range': `bytes */${entry.size}` } });

  const length = end - start + 1;
  const parts = [];
  for (const shard of entry.shards) {
    const sEnd = shard.offset + shard.size - 1;
    if (sEnd < start) continue;
    if (shard.offset > end) break;
    parts.push({
      file: shard.file,
      subStart: Math.max(start, shard.offset) - shard.offset,
      subEnd: Math.min(end, sEnd) - shard.offset,
      shardSize: shard.size,
    });
  }
  if (!parts.length)
    return new Response(null, { status: 416, headers: { 'Content-Range': `bytes */${entry.size}` } });

  const buffers = [];
  for (const p of parts) {
    buffers.push(p.subStart === 0 && p.subEnd === p.shardSize - 1
      ? await fetchShard(p.file, source)
      : await fetchShardSlice(p.file, source, p.subStart, p.subEnd));
  }
  recordProgress(source.pathPrefix, relPath, length);
  scheduleProgress(source.pathPrefix);

  return new Response(new Blob(buffers), {
    status: 206,
    headers: {
      'Content-Type': 'application/octet-stream',
      'Content-Range': `bytes ${start}-${end}/${entry.size}`,
      'Content-Length': String(length),
      'Accept-Ranges': 'bytes',
      'X-Model-SW': 'range',
    },
  });
}

// ─── Shard fetching (with in-flight deduplication) ──────────────────────────
//
// Without dedup: if wllama probes the file (GET/HEAD), then downloads it,
// both requests hit fetchShard concurrently → two CDN fetches for same shard.
// Fix: shardInflight map holds promises for in-progress fetches.

async function fetchShard(file, source) {
  const url = `${source.cdnBase}/${file}`;
  // 1. Check Cache Storage
  try {
    const c = await caches.open(CACHE_NAME);
    const hit = await c.match(url);
    if (hit) return hit.arrayBuffer();
  } catch (_) {}
  // 2. Join existing in-flight fetch if one exists
  if (shardInflight.has(url)) {
    // Return a copy — the original ArrayBuffer may be detached
    const buf = await shardInflight.get(url);
    console.debug(`[model-sw] dedup hit: ${file}`);
    return buf.slice(0);
  }
  // 3. Start new fetch, store promise in dedup map
  const promise = (async () => {
    const r = await fetch(url, { mode: 'cors' });
    if (!r.ok) throw new Error(`[model-sw] ${r.status} ${url}`);
    const buf = await r.arrayBuffer();
    // Write to Cache Storage (fire-and-forget, but with await for safety)
    try {
      const c = await caches.open(CACHE_NAME);
      await c.put(url, new Response(buf.slice(0)));
    } catch (_) {}
    return buf;
  })();
  shardInflight.set(url, promise);
  try {
    return await promise;
  } finally {
    shardInflight.delete(url);
  }
}

async function fetchShardSlice(file, source, sub0, sub1) {
  const url = `${source.cdnBase}/${file}`;
  // Check cache for full shard
  try {
    const c = await caches.open(CACHE_NAME);
    const hit = await c.match(url);
    if (hit) { const full = await hit.arrayBuffer(); return full.slice(sub0, sub1 + 1); }
  } catch (_) {}
  // Try Range request directly (some CDNs support it)
  try {
    const r = await fetch(url, { mode: 'cors', headers: { 'Range': `bytes=${sub0}-${sub1}` } });
    if (r.status === 206) return r.arrayBuffer();
    if (r.ok) {
      // CDN returned full file instead of range — cache it, return slice
      const full = await r.arrayBuffer();
      try {
        const c = await caches.open(CACHE_NAME);
        await c.put(url, new Response(full.slice(0)));
      } catch (_) {}
      return full.slice(sub0, sub1 + 1);
    }
  } catch (_) {}
  // Fallback: fetch full shard via dedup-aware fetchShard
  return (await fetchShard(file, source)).slice(sub0, sub1 + 1);
}

// ─── Filemap loading ────────────────────────────────────────────────────────

async function loadFilemap(cdnBase) {
  if (filemaps.has(cdnBase)) return filemaps.get(cdnBase);
  if (!filemapLoading.has(cdnBase)) {
    filemapLoading.set(cdnBase, (async () => {
      try {
        const r = await fetch(`${cdnBase}/filemap.json`, { mode: 'cors' });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        filemaps.set(cdnBase, data);
        initProgressFromFilemap(cdnBase, data);
        return data;
      } catch (err) {
        console.error(`[model-sw] Filemap failed:`, err);
        filemapLoading.delete(cdnBase);
        return null;
      }
    })());
  }
  return filemapLoading.get(cdnBase);
}
