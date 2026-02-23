#!/usr/bin/env node
/**
 * model-downloader.mjs — Download model files from a filemap to a local folder.
 *
 * Given a filemap.json URL (or local path), downloads files to a target directory.
 * Supports filtering by manifest name(s), SHA256 verification, and resume.
 *
 * Usage:
 *   # Download all files from a CDN filemap:
 *   node model-downloader.mjs \
 *     https://cdn.jsdelivr.net/gh/user/models@v1/filemap.json \
 *     -o ./my-model
 *
 *   # Download only files in the "q4f16" manifest:
 *   node model-downloader.mjs \
 *     https://cdn.jsdelivr.net/gh/user/models@v1/filemap.json \
 *     -m q4f16 -o ./my-model
 *
 *   # Download files from multiple manifests:
 *   node model-downloader.mjs \
 *     https://cdn.jsdelivr.net/gh/user/models@v1/filemap.json \
 *     -m q4f16 -m quantized -o ./my-model
 *
 *   # Download from a local flat-repo (reassemble shards):
 *   node model-downloader.mjs /path/to/pkg-model -o ./my-model
 *
 *   # List available manifests:
 *   node model-downloader.mjs https://example.com/filemap.json --list
 */

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { pipeline } from 'stream/promises';
import { Readable } from 'stream';

// ─── Arg parsing ───────────────────────────────────────────────────────────

const argv = process.argv.slice(2);
const flags = { manifests: [], output: '.', verify: true, list: false, concurrency: 4 };
let source = null;

for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === '-o' || a === '--output')       { flags.output = argv[++i]; }
  else if (a === '-m' || a === '--manifest') { flags.manifests.push(argv[++i]); }
  else if (a === '--list')                   { flags.list = true; }
  else if (a === '--no-verify')              { flags.verify = false; }
  else if (a === '--concurrency')            { flags.concurrency = parseInt(argv[++i], 10); }
  else if (!a.startsWith('-'))               { source = a; }
}

if (!source) {
  console.log('Usage: node model-downloader.mjs <source> [options]');
  console.log('  <source>    URL or local path to filemap.json or directory');
  console.log('  -o DIR      Output directory (default: .)');
  console.log('  -m NAME     Manifest name (repeatable; omit for all files)');
  console.log('  --list      List manifests and exit');
  console.log('  --no-verify Skip SHA256 verification');
  process.exit(1);
}

// ─── Helpers ───────────────────────────────────────────────────────────────

const isUrl = (s) => s.startsWith('http://') || s.startsWith('https://');

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status} fetching ${url}`);
  return r.json();
}

async function loadFilemap(src) {
  if (isUrl(src)) {
    const base = src.endsWith('/filemap.json') ? src.replace(/\/filemap\.json$/, '') : src.replace(/\/$/, '');
    const url = base + '/filemap.json';
    return { filemap: await fetchJson(url), base, remote: true };
  }
  // Local
  let fmapPath;
  if (fs.statSync(src).isDirectory()) {
    fmapPath = path.join(src, 'filemap.json');
  } else {
    fmapPath = src;
  }
  return {
    filemap: JSON.parse(fs.readFileSync(fmapPath, 'utf8')),
    base: path.dirname(fmapPath),
    remote: false,
  };
}

function getFileList(filemap, manifests) {
  const allFiles = new Set(Object.keys(filemap.files || {}));
  if (!manifests.length) return [...allFiles].sort();

  const result = new Set();
  const available = filemap.manifests || {};
  for (const m of manifests) {
    if (!(m in available)) {
      console.log(`  ⚠ Manifest '${m}' not found. Available: ${Object.keys(available).join(', ')}`);
      continue;
    }
    for (const f of available[m].files || []) result.add(f);
  }
  return [...result].sort();
}

function mkdirp(filepath) {
  const dir = path.dirname(filepath);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function cdnName(entry) {
  return entry.cdn_file || entry.cdn || entry.cdnFilename;
}
function shardName(shard) {
  return shard.file || shard.cdn_file || shard.cdn;
}

async function downloadCdn(baseUrl, entry, dest) {
  mkdirp(dest);
  if (fs.existsSync(dest) && fs.statSync(dest).size === entry.size) return 'cached';

  const shards = entry.shards;
  if (!shards) {
    const cdn = cdnName(entry);
    const r = await fetch(`${baseUrl}/${cdn}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    await pipeline(Readable.fromWeb(r.body), fs.createWriteStream(dest));
  } else {
    const ws = fs.createWriteStream(dest);
    for (const shard of shards) {
      const r = await fetch(`${baseUrl}/${shardName(shard)}`);
      if (!r.ok) throw new Error(`HTTP ${r.status} for shard ${shardName(shard)}`);
      await pipeline(Readable.fromWeb(r.body), ws, { end: false });
    }
    ws.end();
    await new Promise(resolve => ws.on('finish', resolve));
  }
  return 'downloaded';
}

function assembleLocal(basePath, entry, dest) {
  mkdirp(dest);
  if (fs.existsSync(dest) && fs.statSync(dest).size === entry.size) return 'cached';

  const shards = entry.shards;
  if (!shards) {
    const cdn = cdnName(entry);
    if (!cdn) return 'skip';
    fs.copyFileSync(path.join(basePath, cdn), dest);
  } else {
    const ws = fs.createWriteStream(dest);
    for (const shard of shards) {
      const data = fs.readFileSync(path.join(basePath, shardName(shard)));
      ws.write(data);
    }
    ws.end();
  }
  return 'assembled';
}

function verifySha256(filepath, expected) {
  const h = crypto.createHash('sha256');
  const data = fs.readFileSync(filepath);
  h.update(data);
  return h.digest('hex') === expected;
}

// ─── Main ──────────────────────────────────────────────────────────────────

async function main() {
  console.log(`Loading filemap from: ${source}`);
  const { filemap, base, remote } = await loadFilemap(source);

  const totalFiles = Object.keys(filemap.files || {}).length;
  const manifests = filemap.manifests || {};
  console.log(`  Filemap v${filemap.version || '?'}: ${totalFiles} files, ${Object.keys(manifests).length} manifest(s)`);

  // List mode
  if (flags.list) {
    if (!Object.keys(manifests).length) {
      console.log('  (no manifests defined)');
    }
    for (const [name, mf] of Object.entries(manifests)) {
      const files = mf.files || [];
      const total = files.reduce((s, f) => s + (filemap.files[f]?.size || 0), 0);
      console.log(`  • ${name}: ${files.length} files, ${(total / 1048576).toFixed(1)} MB`);
    }
    return;
  }

  // File list
  const fileList = getFileList(filemap, flags.manifests);
  if (!fileList.length) { console.log('No files to download.'); return; }

  const totalSize = fileList.reduce((s, f) => s + (filemap.files[f]?.size || 0), 0);
  const label = flags.manifests.length ? flags.manifests.join(', ') : 'all';
  console.log(`  Downloading [${label}]: ${fileList.length} files, ${(totalSize / 1048576).toFixed(1)} MB`);

  const outDir = path.resolve(flags.output);
  fs.mkdirSync(outDir, { recursive: true });

  const stats = { cached: 0, downloaded: 0, assembled: 0, failed: 0, verified: 0 };

  // Process files
  for (let i = 0; i < fileList.length; i++) {
    const vp = fileList[i];
    const entry = filemap.files[vp];
    if (!entry) continue;
    const dest = path.join(outDir, vp);
    const sizeMb = (entry.size / 1048576).toFixed(1);

    try {
      const status = remote
        ? await downloadCdn(base, entry, dest)
        : assembleLocal(base, entry, dest);
      stats[status] = (stats[status] || 0) + 1;
      console.log(`  [${i + 1}/${fileList.length}] ${vp} (${sizeMb} MB) — ${status}`);
    } catch (e) {
      stats.failed++;
      console.log(`  ✗ ${vp}: ${e.message}`);
    }
  }

  // Copy filemap.json
  const fmapDest = path.join(outDir, 'filemap.json');
  if (remote) {
    fs.writeFileSync(fmapDest, JSON.stringify(filemap, null, 2));
  } else {
    fs.copyFileSync(path.join(base, 'filemap.json'), fmapDest);
  }
  console.log(`  filemap.json → ${fmapDest}`);

  // Verify
  if (flags.verify) {
    console.log('\nVerifying SHA256 checksums...');
    for (const vp of fileList) {
      const entry = filemap.files[vp];
      const sha = entry?.sha256;
      const dest = path.join(outDir, vp);
      if (sha && fs.existsSync(dest)) {
        if (verifySha256(dest, sha)) {
          stats.verified++;
        } else {
          console.log(`  ✗ SHA256 MISMATCH: ${vp}`);
          stats.failed++;
        }
      }
    }
  }

  // Summary
  console.log(`\n${'='.repeat(60)}`);
  console.log(`✓ Complete: ${stats.downloaded + stats.assembled} new, ${stats.cached} cached, ${stats.verified} verified, ${stats.failed} failed`);
  console.log(`  Output: ${outDir}`);
  console.log('='.repeat(60));
}

main().catch(e => { console.error('FATAL:', e.message); process.exit(1); });
