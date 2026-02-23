#!/usr/bin/env node
/**
 * Test suite for model-resolver-node.mjs — validates local flat-repo support.
 */

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { ModelResolver, resolveModel } from '/home/claude/WebModelDelivery/model-resolver-node.mjs';

const PKG_EMBEDDING = '/home/claude/obtained/pkg-embedding';
const PKG_RERANKER  = '/home/claude/obtained/pkg-reranker';
const PKG_GEMMA3    = '/home/claude/obtained/pkg-gemma3';
const CACHE_DIR     = '/tmp/test-resolver-cache-node';

let passed = 0, failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`  ✓ ${name}`);
    passed++;
  } catch (e) {
    console.log(`  ✗ ${name}: ${e.message}`);
    failed++;
  }
}

async function testAsync(name, fn) {
  try {
    await fn();
    console.log(`  ✓ ${name}`);
    passed++;
  } catch (e) {
    console.log(`  ✗ ${name}: ${e.message}`);
    failed++;
  }
}

function assertEq(a, b, msg = '') {
  if (a !== b) throw new Error(`${msg}: ${a} !== ${b}`);
}

function assertTrue(v, msg = '') {
  if (!v) throw new Error(msg || 'assertion failed');
}

// ─── Clean start ─────────────────────────────────────────────────────────

if (fs.existsSync(CACHE_DIR)) fs.rmSync(CACHE_DIR, { recursive: true });

const resolver = new ModelResolver({ cacheDir: CACHE_DIR, verifySha256: true });

async function main() {

  // ─── Filemap loading ──────────────────────────────────────────────────

  console.log('\n=== Filemap Loading ===');

  await testAsync('Load embedding filemap from local path', async () => {
    const fm = await resolver._loadFilemap(PKG_EMBEDDING, true);
    assertTrue(fm !== null, 'filemap is null');
    assertEq(fm.version, 5, 'version');
    assertTrue(Object.keys(fm.files).length === 10, `expected 10 files`);
  });

  await testAsync('Load reranker filemap from local path', async () => {
    const fm = await resolver._loadFilemap(PKG_RERANKER, true);
    assertTrue(fm !== null);
    assertTrue('onnx/model_quantized.onnx' in fm.files);
  });

  await testAsync('Load GGUF filemap from local path', async () => {
    const fm = await resolver._loadFilemap(PKG_GEMMA3, true);
    assertTrue(fm !== null);
    assertTrue('gemma-3-1b-it-q4_0.gguf' in fm.files);
  });

  // ─── Resolve: ONNX Embedding ─────────────────────────────────────────

  console.log('\n=== Resolve: ONNX Embedding (q4f16) ===');

  const progressLog = [];

  await testAsync('Resolve embedding to local dir', async () => {
    progressLog.length = 0;
    const localDir = await resolver.resolve(PKG_EMBEDDING, {
      manifest: 'q4f16',
      onProgress: (p) => progressLog.push(p),
    });
    assertTrue(fs.existsSync(localDir), `not a dir: ${localDir}`);
    const fm = await resolver._loadFilemap(PKG_EMBEDDING, true);
    for (const vp of fm.manifests.q4f16.files) {
      const fp = path.join(localDir, vp);
      assertTrue(fs.existsSync(fp), `missing: ${vp}`);
      assertEq(fs.statSync(fp).size, fm.files[vp].size, `size mismatch: ${vp}`);
    }
  });

  await testAsync('Progress tracking (embedding)', async () => {
    assertTrue(progressLog.length > 0, 'no progress events');
    const last = progressLog[progressLog.length - 1];
    assertEq(last.done, true, 'last event not done');
    assertEq(last.percent, 100, 'last event not 100%');
    const pcts = progressLog.map(p => p.percent);
    for (let i = 1; i < pcts.length; i++) {
      assertTrue(pcts[i] >= pcts[i-1], `non-monotonic at ${i}`);
    }
  });

  // ─── Resolve: ONNX Reranker ──────────────────────────────────────────

  console.log('\n=== Resolve: ONNX Reranker (quantized) ===');

  await testAsync('Resolve reranker to local dir', async () => {
    const localDir = await resolver.resolve(PKG_RERANKER, { manifest: 'quantized' });
    const fm = await resolver._loadFilemap(PKG_RERANKER, true);
    const onnxPath = path.join(localDir, 'onnx/model_quantized.onnx');
    assertTrue(fs.existsSync(onnxPath), 'onnx model missing');
    assertEq(fs.statSync(onnxPath).size, fm.files['onnx/model_quantized.onnx'].size);
  });

  // ─── Resolve: GGUF ───────────────────────────────────────────────────

  console.log('\n=== Resolve: GGUF LLM (q4_0) ===');

  await testAsync('Resolve GGUF files', async () => {
    const files = await resolver.resolveFiles(PKG_GEMMA3, { manifest: 'q4_0' });
    const ggufFiles = Object.entries(files).filter(([vp]) => vp.endsWith('.gguf'));
    assertTrue(ggufFiles.length >= 1, 'no gguf files');
    for (const [vp, fp] of ggufFiles) {
      assertTrue(fs.existsSync(fp), `missing: ${fp}`);
      assertTrue(fs.statSync(fp).size > 0, `empty: ${fp}`);
    }
  });

  // ─── SHA256 verification ─────────────────────────────────────────────

  console.log('\n=== SHA256 Verification ===');

  await testAsync('SHA256 verification on resolve', async () => {
    const vr = new ModelResolver({ cacheDir: CACHE_DIR + '-sha', verifySha256: true });
    const localDir = await vr.resolve(PKG_EMBEDDING, { manifest: 'q4f16' });
    const fm = await vr._loadFilemap(PKG_EMBEDDING, true);
    const configEntry = fm.files['config.json'];
    const fp = path.join(localDir, 'config.json');
    const h = crypto.createHash('sha256').update(fs.readFileSync(fp)).digest('hex');
    assertEq(h, configEntry.sha256, 'SHA256 mismatch');
    fs.rmSync(CACHE_DIR + '-sha', { recursive: true });
  });

  // ─── Fetch hook test ─────────────────────────────────────────────────

  console.log('\n=== Fetch Hook (local source) ===');

  await testAsync('Fetch hook intercepts and reassembles', async () => {
    const hookResolver = new ModelResolver({ cacheDir: CACHE_DIR + '-hook' });
    hookResolver.addSource({
      pathPrefix: '/models/embedding/',
      localBase: PKG_EMBEDDING,
      manifest: 'q4f16',
    });
    hookResolver.installFetchHook();

    // Fetch a small file through the hook
    const resp = await fetch('/models/embedding/config.json');
    assertTrue(resp.ok, `fetch failed: ${resp.status}`);
    const text = await resp.text();
    const parsed = JSON.parse(text);
    assertTrue('model_type' in parsed || 'architectures' in parsed || Object.keys(parsed).length > 0,
      'config.json not valid JSON model config');

    // Fetch a sharded file
    const resp2 = await fetch('/models/embedding/onnx/model_q4f16.onnx_data');
    assertTrue(resp2.ok, `fetch sharded failed: ${resp2.status}`);
    const fm = await hookResolver._loadFilemap(PKG_EMBEDDING, true);
    assertEq(parseInt(resp2.headers.get('Content-Length')), fm.files['onnx/model_q4f16.onnx_data'].size,
      'content-length mismatch for sharded file');

    // Range request test
    const resp3 = await fetch('/models/embedding/onnx/model_q4f16.onnx_data', {
      headers: { Range: 'bytes=0-99' },
    });
    assertEq(resp3.status, 206, 'expected 206 for range');
    const rangeBody = await resp3.arrayBuffer();
    assertEq(rangeBody.byteLength, 100, 'range body length');

    hookResolver.removeFetchHook();
    if (fs.existsSync(CACHE_DIR + '-hook')) fs.rmSync(CACHE_DIR + '-hook', { recursive: true });
  });

  // ─── Cache skip ──────────────────────────────────────────────────────

  console.log('\n=== Cache Skip ===');

  await testAsync('Second resolve skips (cached)', async () => {
    const t0 = Date.now();
    await resolver.resolve(PKG_EMBEDDING, { manifest: 'q4f16' });
    const dt = (Date.now() - t0) / 1000;
    assertTrue(dt < 2.0, `second resolve too slow: ${dt.toFixed(1)}s`);
  });

  // ─── Summary ─────────────────────────────────────────────────────────

  console.log(`\n${'='.repeat(50)}`);
  console.log(`Node.js resolver tests: ${passed} passed, ${failed} failed`);
  if (failed) {
    process.exit(1);
  } else {
    console.log('All tests passed ✓');
  }
}

main().catch(e => { console.error('FATAL:', e); process.exit(1); });
