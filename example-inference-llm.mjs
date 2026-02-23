#!/usr/bin/env node
/**
 * example-inference-llm.mjs — End-to-end GGUF LLM inference via model-resolver-node
 *
 * Demonstrates:
 *   1. Resolving a GGUF model from a local flat-repo (or CDN)
 *   2. Loading with node-llama-cpp (npm install node-llama-cpp)
 *   3. Single-turn and multi-turn chat with LlamaChatSession
 *
 * Usage:
 *   node example-inference-llm.mjs --source /path/to/pkg-gemma3
 *
 *   # CDN source:
 *   node example-inference-llm.mjs \
 *     --source https://cdn.jsdelivr.net/gh/user/cdn-llm@v1 \
 *     --manifest q4_0
 *
 * Requirements:
 *   npm install node-llama-cpp
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ─── Parse args ────────────────────────────────────────────────────────────

const argv = process.argv.slice(2);
const getArg = (name, fallback) => {
  const i = argv.indexOf(name);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback;
};

const SOURCE   = getArg('--source', null);
const MANIFEST = getArg('--manifest', 'q4_0');
const CACHE_DIR = getArg('--cache-dir', './.model-cache');
const MAX_TOKENS = parseInt(getArg('--max-tokens', '128'), 10);

if (!SOURCE) {
  console.log('Usage: node example-inference-llm.mjs --source /path/to/pkg-gemma3');
  process.exit(1);
}

// ─── Main ──────────────────────────────────────────────────────────────────

async function main() {
  const { ModelResolver } = await import(path.join(__dirname, 'model-resolver-node.mjs'));
  const { getLlama, LlamaChatSession } = await import('node-llama-cpp');

  const resolver = new ModelResolver({ cacheDir: CACHE_DIR });

  console.log('='.repeat(60));
  console.log('LLM INFERENCE via model-resolver-node + node-llama-cpp');
  console.log('='.repeat(60));

  // ── 1. Resolve GGUF model ─────────────────────────────────────────────

  console.log(`\nResolving GGUF model from: ${SOURCE}`);
  let t0 = Date.now();
  const files = await resolver.resolveFiles(SOURCE, {
    manifest: MANIFEST,
    onProgress: (p) => process.stdout.write(
      `\r  [${String(p.percent).padStart(3)}%] ${(p.file || '').slice(0, 50).padEnd(50)}`),
  });

  const ggufPath = Object.entries(files)
    .filter(([vp]) => vp.endsWith('.gguf'))
    .map(([, fp]) => fp)[0];

  if (!ggufPath) { console.error('No .gguf file found in manifest'); process.exit(1); }

  const sizeMb = (fs.statSync(ggufPath).size / 1048576).toFixed(1);
  console.log(`\n  → ${ggufPath}  (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
  console.log(`  Size: ${sizeMb} MB`);

  // ── 2. Load model ─────────────────────────────────────────────────────

  console.log('\nLoading model with node-llama-cpp...');
  t0 = Date.now();
  const llama = await getLlama({ gpu: false });
  const model = await llama.loadModel({ modelPath: ggufPath });
  console.log(`  ✓ Model loaded (${((Date.now() - t0) / 1000).toFixed(1)}s)`);

  // ── 3. Single-turn inference ──────────────────────────────────────────

  console.log('\n' + '='.repeat(60));
  console.log('SINGLE-TURN INFERENCE');
  console.log('-'.repeat(60));

  const prompts = [
    'What is 2 + 2? Answer in one sentence.',
    'Explain gravity to a 5 year old in two sentences.',
    'Write a haiku about programming.',
  ];

  for (const prompt of prompts) {
    // Fresh context+session per prompt for single-turn
    const sCtx = await model.createContext({ contextSize: 512 });
    const session = new LlamaChatSession({ contextSequence: sCtx.getSequence() });
    console.log(`\n  User: ${prompt}`);
    t0 = Date.now();
    const response = await session.prompt(prompt, { maxTokens: MAX_TOKENS, temperature: 0.7 });
    console.log(`  Model: ${response.trim()}`);
    console.log(`  (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
    await sCtx.dispose();
  }

  // ── 4. Multi-turn conversation ────────────────────────────────────────

  console.log('\n' + '='.repeat(60));
  console.log('MULTI-TURN CONVERSATION');
  console.log('-'.repeat(60));

  // node-llama-cpp LlamaChatSession maintains history automatically
  const chatCtx = await model.createContext({ contextSize: 2048 });
  const chat = new LlamaChatSession({ contextSequence: chatCtx.getSequence() });

  const turns = [
    'What is the capital of France?',
    'What is its population?',
    'And what is a famous landmark there?',
  ];

  for (const turn of turns) {
    console.log(`\n  User: ${turn}`);
    const reply = await chat.prompt(turn, { maxTokens: 64, temperature: 0.3 });
    console.log(`  Model: ${reply.trim()}`);
  }

  // ── Cleanup ───────────────────────────────────────────────────────────

  await chatCtx.dispose();
  await model.dispose();
  await llama.dispose();

  console.log('\n' + '='.repeat(60));
  console.log('✓ All LLM inference complete');
  console.log('='.repeat(60));
}

main().catch(e => { console.error('FATAL:', e.message); process.exit(1); });
