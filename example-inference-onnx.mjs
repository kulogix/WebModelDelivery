#!/usr/bin/env node
/**
 * example-inference-onnx.mjs — End-to-end ONNX inference via model-resolver-node
 *
 * Demonstrates:
 *   1. Resolving ONNX models from local flat-repo (or CDN) via the Node.js resolver
 *   2. Tokenization with @huggingface/transformers AutoTokenizer
 *   3. Embedding inference — computing sentence embeddings + cosine similarity
 *   4. Reranker inference — scoring query–document relevance
 *   5. Fetch-hook interception — transparent shard reassembly
 *
 * Usage:
 *   node example-inference-onnx.mjs \
 *     --embedding-source /path/to/pkg-embedding \
 *     --reranker-source /path/to/pkg-reranker
 *
 *   # CDN sources:
 *   node example-inference-onnx.mjs \
 *     --embedding-source https://cdn.jsdelivr.net/gh/user/cdn-embedding@v1 \
 *     --reranker-source https://cdn.jsdelivr.net/gh/user/cdn-reranker@v1
 *
 * Requirements:
 *   npm install @huggingface/transformers   (includes onnxruntime-node)
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ─── Parse args ────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const getArg = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};

const EMBEDDING_SOURCE  = getArg('--embedding-source', null);
const EMBEDDING_MANIFEST = getArg('--embedding-manifest', 'q4f16');
const RERANKER_SOURCE   = getArg('--reranker-source', null);
const RERANKER_MANIFEST  = getArg('--reranker-manifest', 'quantized');
const CACHE_DIR = getArg('--cache-dir', './.model-cache');

if (!EMBEDDING_SOURCE || !RERANKER_SOURCE) {
  console.log('Usage: node example-inference-onnx.mjs \\');
  console.log('  --embedding-source /path/to/pkg-embedding \\');
  console.log('  --reranker-source /path/to/pkg-reranker');
  process.exit(1);
}

// ─── Imports ───────────────────────────────────────────────────────────────

const ort = await import('onnxruntime-node');
const { AutoTokenizer } = await import('@huggingface/transformers');
const { ModelResolver } = await import(path.join(__dirname, 'model-resolver-node.mjs'));

// ─── Helpers ───────────────────────────────────────────────────────────────

function cosineSimilarity(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function toOrtTensor(arr) {
  return new ort.Tensor('int64', BigInt64Array.from(arr.map(BigInt)), [1, arr.length]);
}

function progress(p) {
  process.stdout.write(`\r  [${String(p.percent).padStart(3)}%] ${(p.file || '').slice(0, 50).padEnd(50)}`);
}

// ─── Main ──────────────────────────────────────────────────────────────────

async function main() {
  const resolver = new ModelResolver({ cacheDir: CACHE_DIR, verifySha256: true });

  console.log('='.repeat(60));
  console.log('ONNX INFERENCE via model-resolver-node + onnxruntime-node');
  console.log('='.repeat(60));

  // ── 1. Resolve models ─────────────────────────────────────────────────

  console.log('\nResolving embedding model...');
  let t0 = Date.now();
  const embDir = await resolver.resolve(EMBEDDING_SOURCE, {
    manifest: EMBEDDING_MANIFEST, onProgress: progress,
  });
  console.log(`\n  → ${embDir}  (${((Date.now() - t0) / 1000).toFixed(1)}s)`);

  console.log('\nResolving reranker model...');
  t0 = Date.now();
  const rrDir = await resolver.resolve(RERANKER_SOURCE, {
    manifest: RERANKER_MANIFEST, onProgress: progress,
  });
  console.log(`\n  → ${rrDir}  (${((Date.now() - t0) / 1000).toFixed(1)}s)`);

  // ── 2. Load tokenizers + ONNX sessions ────────────────────────────────

  console.log('\n' + '='.repeat(60));
  console.log('Loading tokenizers and ONNX sessions...');

  const embTok  = await AutoTokenizer.from_pretrained(embDir, { local_files_only: true });
  const rrTok   = await AutoTokenizer.from_pretrained(rrDir, { local_files_only: true });
  const embSess = await ort.InferenceSession.create(path.join(embDir, 'onnx', 'model_q4f16.onnx'));
  const rrSess  = await ort.InferenceSession.create(path.join(rrDir, 'onnx', 'model_quantized.onnx'));

  console.log(`  ✓ Embedding: [${embSess.inputNames}] → [${embSess.outputNames}]`);
  console.log(`  ✓ Reranker:  [${rrSess.inputNames}] → [${rrSess.outputNames}]`);

  // ── 3. Embedding inference ────────────────────────────────────────────

  console.log('\n' + '='.repeat(60));
  console.log('EMBEDDING INFERENCE');
  console.log('-'.repeat(60));

  const sentences = [
    'The quick brown fox jumps over the lazy dog.',
    'A fast auburn fox leaps above a sleepy hound.',
    'Machine learning models can run in the browser.',
    'The stock market closed higher on Friday.',
  ];

  const embeddings = [];
  for (const sent of sentences) {
    const enc = embTok(sent);
    const ids  = Array.from(enc.input_ids.data, Number);
    const mask = Array.from(enc.attention_mask.data, Number);

    t0 = Date.now();
    const out = await embSess.run({
      input_ids: toOrtTensor(ids),
      attention_mask: toOrtTensor(mask),
    });
    const dt = Date.now() - t0;

    const emb = Array.from(out.sentence_embedding.data);
    embeddings.push(emb);
    const norm = Math.sqrt(emb.reduce((s, x) => s + x * x, 0));
    console.log(`  "${sent.slice(0, 50)}"`);
    console.log(`    → dim=${emb.length}, norm=${norm.toFixed(4)}, ${dt}ms`);
  }

  console.log('\nCosine similarities:');
  for (let i = 0; i < sentences.length; i++) {
    for (let j = i + 1; j < sentences.length; j++) {
      const sim = cosineSimilarity(embeddings[i], embeddings[j]);
      const tag = sim > 0.7 ? ' ← similar!' : '';
      console.log(`  [${i}] vs [${j}]: ${sim.toFixed(4)}${tag}`);
    }
  }

  // ── 4. Reranker inference ─────────────────────────────────────────────

  console.log('\n' + '='.repeat(60));
  console.log('RERANKER INFERENCE');
  console.log('-'.repeat(60));

  const query = 'How do neural networks learn?';
  const documents = [
    'Neural networks adjust weights through backpropagation during training.',
    'The stock market experienced volatility this quarter.',
    'Deep learning uses gradient descent to minimize loss functions.',
    'Photosynthesis converts sunlight into chemical energy in plants.',
    'Transformers use self-attention mechanisms for sequence modeling.',
  ];

  console.log(`  Query: "${query}"`);
  console.log(`  Documents: ${documents.length}`);

  const scores = [];
  for (const doc of documents) {
    const enc = rrTok(query, { text_pair: doc });
    const ids  = Array.from(enc.input_ids.data, Number);
    const mask = Array.from(enc.attention_mask.data, Number);

    const out = await rrSess.run({
      input_ids: toOrtTensor(ids),
      attention_mask: toOrtTensor(mask),
    });
    scores.push(Array.from(out.logits.data)[0]);
  }

  const ranked = documents
    .map((doc, i) => ({ doc, score: scores[i], i }))
    .sort((a, b) => b.score - a.score);

  console.log('\n  Ranked results:');
  for (const [rank, { doc, score }] of ranked.entries()) {
    console.log(`    #${rank + 1} (score: ${score >= 0 ? '+' : ''}${score.toFixed(4)}): "${doc.slice(0, 70)}"`);
  }

  // ── 5. Fetch-hook demonstration ───────────────────────────────────────

  console.log('\n' + '='.repeat(60));
  console.log('FETCH HOOK DEMONSTRATION');
  console.log('-'.repeat(60));

  const hookResolver = new ModelResolver({ cacheDir: CACHE_DIR + '-hook' });
  hookResolver.addSource({
    pathPrefix: '/models/embedding/',
    localBase: EMBEDDING_SOURCE,
    manifest: EMBEDDING_MANIFEST,
  });
  hookResolver.installFetchHook();

  const r1 = await fetch('/models/embedding/config.json');
  const cfg = await r1.json();
  console.log(`  fetch config.json → ${r1.status}, model_type: ${cfg.model_type}`);

  const r2 = await fetch('/models/embedding/onnx/model_q4f16.onnx_data');
  console.log(`  fetch sharded onnx_data → ${r2.status}, size: ${r2.headers.get('Content-Length')}`);

  const r3 = await fetch('/models/embedding/onnx/model_q4f16.onnx_data', {
    headers: { Range: 'bytes=0-1023' },
  });
  console.log(`  range request → ${r3.status}, Content-Range: ${r3.headers.get('Content-Range')}`);

  hookResolver.removeFetchHook();

  console.log('\n' + '='.repeat(60));
  console.log('✓ All ONNX inference complete');
  console.log('='.repeat(60));
}

main().catch(e => { console.error('FATAL:', e); process.exit(1); });
