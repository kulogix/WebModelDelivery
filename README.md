# WebModelDelivery — ML Model Packaging & Delivery

Package machine learning models into CDN-friendly shards with transparent
reassembly for browser, Node.js, and Python runtimes.

```
┌────────────┐    ┌──────────────┐    ┌─────────────────┐
│ HuggingFace │    │ model-       │    │  CDN / Local     │
│ model repo  │───▶│ packager.sh  │───▶│  flat-repo       │
│             │    │              │    │  (filemap.json)  │
└────────────┘    └──────────────┘    └────────┬────────┘
                                               │
         ┌──────────────────┬──────────────────┤
         ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ model-sw.js  │  │ model-resolver-  │  │ model_resolver.py│
│ (Browser SW) │  │ node.mjs         │  │ (Python)         │
│              │  │                  │  │                  │
│ fetch() hook │  │ fetch() hook     │  │ resolve() API    │
│ → reassemble │  │ → reassemble     │  │ → reassemble     │
└──────┬───────┘  └──────┬───────────┘  └──────┬───────────┘
       ▼                 ▼                     ▼
  Transformers.js   onnxruntime-node      onnxruntime
  wllama            node-llama-cpp        llama-cpp-python
```

## Why?

**Problem 1: CDN file size limits.** JSDelivr caps individual files at 20 MB.
A typical ONNX model weighs 90–2000 MB. GitHub raw serving is unreliable for
large files. HuggingFace Hub may be blocked in enterprise/education networks.

**Problem 2: Multi-quantization repos.** A single model repo might contain fp32,
fp16, q8, and q4 variants — 2+ GB total. Your app loads one variant (~200 MB).
Without manifests, progress tracking reports 200 MB / 2 GB = 10%.

**Problem 3: CDN hosting is simple.** Git push to GitHub, toggle JSDelivr, done.
No special infrastructure, no CORS configuration, no storage bills beyond GitHub's
free tier. But the model files need to be chunked first.

**Solution:** `model-packager.sh` splits model files into CDN-safe shards (<19 MB),
generates a `filemap.json` with reassembly instructions, and `model-sw.js` intercepts
browser fetches and transparently reassembles them. Your ML framework (Transformers.js,
wllama, etc.) has no idea it's reading from shards.

---

## Quick Start

```bash
# 1. Clone and package a model
git clone https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX
./model-packager.sh -o ./cdn-embedding ./embeddinggemma-300m-ONNX/

# 2. Push to GitHub
cd cdn-embedding && git init && git add . && git commit -m "initial"
git remote add origin https://github.com/you/cdn-embedding.git
git push -u origin main

# 3. Use in your app (3 lines)
# Register model-sw.js, point it at JSDelivr, load with Transformers.js
```

---

## Table of Contents

1.  [Installation & Dependencies](#installation--dependencies)
2.  [model-packager.sh — Packaging Models](#model-packagersh)
3.  [model-sw.js — Service Worker](#model-swjs)
4.  [Node.js Resolver — model-resolver-node.mjs](#nodejs-resolver)
5.  [Python Resolver — model_resolver.py](#python-resolver)
6.  [Model Downloader Scripts](#model-downloader-scripts)
7.  [End-to-End Inference Examples](#end-to-end-inference-examples)
8.  [Integration Guide](#integration-guide)
9.  [Progress Tracking](#progress-tracking)
10. [Manifests & Multi-Quantization](#manifests--multi-quantization)
11. [Test Harness (ONNX)](#test-harness)
12. [Test Harness (LLM / wllama)](#test-harness-llm--wllama)
13. [Running Tests (Headless)](#running-tests-headless)
14. [Running Interactively](#running-interactively)
15. [Filemap Format Reference](#filemap-format-reference)
16. [Troubleshooting](#troubleshooting)

---

## Installation & Dependencies

### Packager requirements

```bash
# Required
bash 3.2+     # macOS default works
python3       # For filemap JSON generation
split         # Standard Unix utility (pre-installed on macOS/Linux)

# Optional (GGUF models only)
llama-gguf-split   # From llama.cpp releases — only needed for GGUF re-splitting
```

The packager auto-detects `sha256sum`, `gsha256sum`, or `shasum -a 256` (macOS built-in)
for checksums — no extra installs needed.

The `-s` / `--chunk-size` flag accepts both raw bytes (`20000000`) and human-readable
suffixes (`20M`, `1g`, `512k`).

### Browser/harness requirements

```bash
# Node.js 18+ for the dev server and headless tests
node --version   # v18+ required for native fetch

# NPM packages for the test harness — install what you need:
# ONNX models (Transformers.js):
npm install @huggingface/transformers   # Transformers.js v3
npm install onnxruntime-web             # ONNX Runtime WASM backend

# GGUF / LLM models (wllama):
npm install @wllama/wllama              # llama.cpp WASM bindings

# Headless testing:
npm install puppeteer-core              # Headless Chrome control (tests only)

# Chrome/Chromium for headless tests
# See "Finding Chrome" section below
```

### Inference example requirements (Node.js)

```bash
# ONNX examples (embedding + reranker):
npm install @huggingface/transformers   # includes onnxruntime-node

# LLM examples (GGUF chat):
npm install node-llama-cpp
```

### Inference example requirements (Python)

```bash
# ONNX examples:
pip install onnxruntime tokenizers numpy

# LLM examples:
pip install llama-cpp-python
```

### Finding Chrome for headless tests

The headless test runner needs a Chrome or Chromium binary:

```bash
# Linux (common paths)
/usr/bin/google-chrome
/usr/bin/chromium-browser
/opt/google/chrome/chrome
/snap/bin/chromium

# macOS
/Applications/Google Chrome.app/Contents/MacOS/Google Chrome

# Install if missing
# Ubuntu/Debian:
sudo apt-get install -y chromium-browser
# Or install full Chrome:
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb

# Alternatively, let Puppeteer install its own:
npx puppeteer browsers install chrome
# Then use: node_modules/puppeteer-core/.local-chromium/*/chrome
```

---

## model-packager.sh

### Basic usage

```bash
# Package a folder (auto-discovers all files)
./model-packager.sh -o ./output-dir ./path/to/model/

# Package specific files
./model-packager.sh -o ./output-dir model.onnx tokenizer.json config.json

# GGUF model with custom shard size
./model-packager.sh -o ./output-dir --gguf-shard-size 500M ./my-model.gguf
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output DIR` | (required) | Output directory. Created if needed. Always flat structure. |
| `-s, --chunk-size BYTES` | `19922944` (19 MiB) | Max CDN chunk size. Set to match your CDN's limit. |
| `--merge` | off | Merge into existing output. Shared files (same SHA256) are deduplicated. |
| `--overwrite` | off | Wipe existing output directory before packaging. |
| `--manifest NAME` | (auto-detect) | Explicit manifest name for this run's files. |
| `--gguf-split PATH` | search PATH | Path to `llama-gguf-split` binary. |
| `--gguf-shard-size SIZE` | `1800M` | Max GGUF shard size (must be <2GB for wllama). |
| `--keep-intermediates` | off | Keep full gguf-split shards alongside CDN chunks. |
| `--remove-originals` | off | Delete input files after successful packaging. |
| `--exclude PATTERN` | git/system files | Additional glob pattern to exclude (repeatable). |
| `--dry-run` | off | Show what would happen without writing files. |
| `-v, --verbose` | off | Verbose output. |

### How it works (phases)

1. **Phase 0 — Discovery.** Walks input folders, collects files, checks for GGUF.
2. **Phase 1 — GGUF splitting.** If inputs include `.gguf` files larger than
   `--gguf-shard-size`, splits them with `llama-gguf-split`. Pre-split GGUFs
   (with `-00001-of-00003` naming) are detected automatically.
3. **Phase 2 — SHA256 hashing.** Computes hashes for deduplication in merge mode.
4. **Phase 3 — Byte splitting.** Files >chunk-size are split into numbered shards
   (`filename.shard.000`, `.shard.001`, etc.) with offset metadata.
5. **Phase 4 — Copy to output.** Flat structure. In merge mode, files with matching
   SHA256 are skipped (pointed to existing copy).
6. **Phase 5 — Filemap generation.** Writes `filemap.json` v5 with file entries
   including size, SHA256, shard maps, and CDN filenames.
7. **Phase 5b — Manifest generation.** Auto-detects quantization groups from
   filenames. For GGUF files, extracts metadata (architecture, quant, context length)
   via `gguf-meta.py`, classifies as LLM or mmproj, and generates cross-permutation
   manifests for multimodal models. Or uses `--manifest NAME` if specified.
8. **Phase 6 — Verification.** Checks all CDN files are under the size limit.

### Multi-quantization packaging

```bash
# Method 1: Separate runs with --merge
./model-packager.sh -o ./cdn-pkg models/model-q4/
./model-packager.sh -o ./cdn-pkg --merge models/model-q8/
./model-packager.sh -o ./cdn-pkg --merge models/model-fp16/

# Result: single filemap.json with 3 manifests (q4, q8, fp16)
# Shared files (config.json, tokenizer.json) stored ONCE — SHA256 dedup

# Method 2: Single folder containing all variants
# If models/combined/ has model_q4.onnx, model_q8.onnx, model_fp16.onnx:
./model-packager.sh -o ./cdn-pkg models/combined/
# Auto-detects 3 ONNX groups, generates 3 manifests

# Method 3: Explicit manifest names (for complex combos)
./model-packager.sh -o ./cdn-llm --manifest "Q4_K_M" models/text-Q4_K_M.gguf
./model-packager.sh -o ./cdn-llm --merge --manifest "mmproj-f16" models/mmproj-f16.gguf
```

### CDN chunk size rationale

The default 19 MiB (19,922,944 bytes) provides headroom under JSDelivr's 20 MB limit.
Other CDNs may have different limits:

| CDN | File size limit | Recommended `--chunk-size` |
|-----|----------------|---------------------------|
| JSDelivr | 20 MB | 19922944 (default) |
| GitHub Pages | 100 MB | 99000000 |
| Cloudflare Pages | 25 MB | 24000000 |
| S3 + CloudFront | no limit | single file, no sharding needed |

---

## model-sw.js

The Service Worker intercepts fetch requests matching configured path prefixes,
looks up the virtual path in the filemap, fetches the CDN shards, and reassembles
them into the original file — transparently to the consuming library.

### Minimal integration (no progress)

```html
<script>
  // 1. Register the Service Worker
  navigator.serviceWorker.register('/model-sw.js', { scope: '/' });
  await navigator.serviceWorker.ready;

  // 2. Configure sources
  navigator.serviceWorker.controller.postMessage({
    type: 'MODEL_SW_INIT',
    sources: [{
      pathPrefix: '/models/embedding/',
      cdnBase: 'https://cdn.jsdelivr.net/gh/you/cdn-embedding@v1',
    }]
  });

  // 3. Load as if files were local — SW handles the rest
  const model = await AutoModel.from_pretrained('/models/embedding/', {
    dtype: 'q4f16',
  });
</script>
```

### Source configuration

Each source object in the `MODEL_SW_INIT` message:

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `pathPrefix` | yes | — | URL path prefix to intercept (e.g., `/models/embedding/`) |
| `cdnBase` | yes | — | CDN URL root where `filemap.json` and shards live |
| `progress` | no | `false` | Enable progress tracking (opt-in, zero overhead when false) |
| `manifest` | no | — | Manifest name for progress denominator. Omit for adaptive mode. |

### Messages

| Message | Direction | Description |
|---------|-----------|-------------|
| `MODEL_SW_INIT` | page → SW | Configure sources. Resets all state. |
| `MODEL_SW_PROGRESS` | SW → page | Progress update (only if `progress: true`). |
| `MODEL_SW_COMPLETE` | page → SW | Force 100% for a pathPrefix. Send after model loads. |
| `MODEL_SW_CLEAR_CACHE` | page → SW | Clear the shard cache. |
| `MODEL_SW_CACHE_CLEARED` | SW → page | Confirms cache was cleared. |
| `MODEL_SW_STATUS` | page → SW / SW → page | Request/report current sources and filemap state. |

### How fetch interception works

```
Browser requests: /models/embedding/onnx/model_q4f16.onnx_data
                  ↓
SW matches pathPrefix: /models/embedding/
Relative path: onnx/model_q4f16.onnx_data
                  ↓
Looks up in filemap.json:
  "onnx/model_q4f16.onnx_data": {
    "size": 175234567,
    "shards": [
      {"file": "model_q4f16.onnx_data.shard.000", "offset": 0, "size": 19922944},
      {"file": "model_q4f16.onnx_data.shard.001", "offset": 19922944, "size": 19922944},
      ...
    ]
  }
                  ↓
Fetches each shard from CDN, concatenates into a ReadableStream
Returns single Response with correct Content-Length
                  ↓
Transformers.js sees a normal 175 MB file. No idea about shards.
```

For Range requests (common with ONNX Runtime), the SW maps byte ranges to the
correct shard(s) and returns a proper 206 Partial Content response.

### Caching

Shards are cached in the Cache API (`model-shards-v1`). Second loads are instant.
Clear with `MODEL_SW_CLEAR_CACHE` or manually: `caches.delete('model-shards-v1')`.

---

## Node.js Resolver

`model-resolver-node.mjs` brings the same shard-reassembly logic to Node.js. It can load
models from **local flat-repos** (packaged directories) or **CDN URLs**, with automatic
SHA256 verification, progress tracking, and a transparent `fetch()` hook.

### Installation

```bash
npm install @huggingface/transformers   # includes onnxruntime-node
# model-resolver-node.mjs has zero npm dependencies — just copy it into your project
```

### Basic Usage — resolve to a directory

```js
import { ModelResolver } from './model-resolver-node.mjs';

const resolver = new ModelResolver({ cacheDir: './.model-cache' });

// Local flat-repo (from model-packager.sh output)
const dir = await resolver.resolve('/path/to/pkg-embedding', {
  manifest: 'q4f16',
  onProgress: (p) => console.log(`${p.percent}% ${p.file}`),
});
// dir → '.model-cache/resolved/d3b466c7c7be_q4f16/' with original file structure

// CDN source (identical API)
const dir2 = await resolver.resolve('https://cdn.jsdelivr.net/gh/user/cdn-model@v1', {
  manifest: 'q4f16',
});
```

### Fetch Hook — transparent interception

The fetch hook lets libraries like Transformers.js load models as if they were local,
while the resolver transparently reassembles shards behind the scenes:

```js
const resolver = new ModelResolver({ cacheDir: './.model-cache' });

resolver.addSource({
  pathPrefix: '/models/embedding/',
  localBase: '/path/to/pkg-embedding',  // or CDN URL
  manifest: 'q4f16',
});
resolver.installFetchHook();

// Now fetch() calls are intercepted:
const config = await (await fetch('/models/embedding/config.json')).json();
const onnx   = await (await fetch('/models/embedding/onnx/model.onnx')).arrayBuffer();

// Range requests work too — wllama uses these for GGUF streaming:
const chunk = await fetch('/models/llm/model.gguf', {
  headers: { Range: 'bytes=0-1023' },
});
// → 206 Partial Content, Content-Range: bytes 0-1023/957117312

resolver.removeFetchHook();
```

### Key Methods

| Method | Description |
|--------|-------------|
| `resolve(source, opts)` | Resolve all manifest files to a cache directory |
| `resolveFiles(source, opts)` | Returns `{ virtualPath: absolutePath }` map |
| `addSource(config)` | Register a source for the fetch hook |
| `installFetchHook()` | Monkey-patch `globalThis.fetch` |
| `removeFetchHook()` | Restore original `fetch` |
| `listManifests(source)` | List available manifest names |

---

## Python Resolver

`model_resolver.py` provides the same functionality for Python workflows — load models
from local flat-repos or CDN into a cache directory, then point onnxruntime, llama-cpp-python,
or any other framework at the resolved path.

### Basic Usage

```python
from model_resolver import ModelResolver, resolve_gguf

resolver = ModelResolver(cache_dir='./.model-cache')

# Resolve an ONNX model
model_dir = resolver.resolve(
    '/path/to/pkg-embedding',       # or CDN URL
    manifest='q4f16',
    on_progress=lambda p: print(f"{p['percent']}% {p['file']}"),
)
# model_dir → '.model-cache/resolved/d3b466c7c7be_q4f16/'

import onnxruntime as ort
session = ort.InferenceSession(f'{model_dir}/onnx/model_q4f16.onnx')

# Resolve a GGUF model (convenience function returns list of .gguf paths)
gguf_paths = resolve_gguf('/path/to/pkg-gemma3', manifest='q4_0')
# gguf_paths → ['.model-cache/resolved/ae5120dd/gemma-3-1b-it-q4_0.gguf']
```

### CLI

```bash
# List manifests
python model_resolver.py list /path/to/pkg-embedding

# Resolve to cache
python model_resolver.py resolve /path/to/pkg-embedding --manifest q4f16

# Serve for browser use (development)
python model_resolver.py serve /path/to/pkg-embedding --port 8080
```

---

## Model Downloader Scripts

Standalone scripts to download (or reassemble) model files from a filemap.json source
to a local directory. Useful for pre-fetching models for offline use, CI/CD pipelines,
or migrating between hosting providers.

### Python — `model-downloader.py`

```bash
# Download all files from a CDN filemap
python model-downloader.py \
  https://cdn.jsdelivr.net/gh/user/models@v1/filemap.json \
  -o ./my-model

# Download only the q4f16 manifest
python model-downloader.py \
  https://cdn.jsdelivr.net/gh/user/models@v1/filemap.json \
  -m q4f16 -o ./my-model

# Download multiple manifests
python model-downloader.py /path/to/filemap.json \
  -m q4f16 -m quantized -o ./my-model

# List available manifests
python model-downloader.py https://example.com/filemap.json --list

# Reassemble from a local flat-repo
python model-downloader.py /path/to/pkg-embedding -o ./my-model
```

### Node.js — `model-downloader.mjs`

```bash
# Download all files
node model-downloader.mjs \
  https://cdn.jsdelivr.net/gh/user/models@v1/filemap.json \
  -o ./my-model

# Filter by manifest
node model-downloader.mjs /path/to/filemap.json -m q4f16 -o ./my-model

# Multiple manifests
node model-downloader.mjs /path/to/filemap.json -m q4f16 -m quantized -o ./output

# List manifests
node model-downloader.mjs https://example.com/filemap.json --list
```

### Options

| Flag | Description |
|------|-------------|
| `-o, --output DIR` | Target directory (default: current directory) |
| `-m, --manifest NAME` | Manifest to download (repeatable; omit for all files) |
| `--list` | List available manifests and exit |
| `--no-verify` | Skip SHA256 verification |
| `--concurrency N` | Parallel downloads for CDN (Python only, default: 4) |

The downloaded directory includes `filemap.json` alongside the reassembled original files.
Files are verified against SHA256 checksums in the filemap by default.

---

## End-to-End Inference Examples

Four complete examples demonstrate the full pipeline from filemap → resolved model → inference.

### Python + ONNX — Embedding & Reranker

```bash
pip install onnxruntime tokenizers numpy

python example-inference-onnx.py \
  --embedding-source /path/to/pkg-embedding \
  --reranker-source /path/to/pkg-reranker
```

Output:
```
EMBEDDING INFERENCE
  "The quick brown fox jumps over the lazy dog."
    → dim=768, norm=1.0000, 61ms
  "A fast auburn fox leaps above a sleepy hound."
    → dim=768, norm=1.0000, 30ms

Cosine similarities:
  [0] vs [1]: 0.8099 ← similar!
  [2] vs [3]: 0.1807

RERANKER INFERENCE
  Query: "How do neural networks learn?"
    #1 (score: +0.9591): "Neural networks adjust weights through backpropagation..."
    #2 (score: -0.0123): "Deep learning uses gradient descent..."
    #5 (score: -3.5461): "The stock market experienced volatility..."
```

### Node.js + ONNX — Embedding & Reranker

```bash
npm install @huggingface/transformers  # includes onnxruntime-node

node example-inference-onnx.mjs \
  --embedding-source /path/to/pkg-embedding \
  --reranker-source /path/to/pkg-reranker
```

Produces identical results to the Python example. Also demonstrates the fetch hook
(transparent shard reassembly through intercepted `fetch()` calls) and Range request handling.

### Python + LLM — GGUF Chat with llama-cpp-python

```bash
pip install llama-cpp-python

python example-inference-llm.py \
  --source /path/to/pkg-gemma3 \
  --max-tokens 128
```

Output:
```
SINGLE-TURN INFERENCE
  User: What is 2 + 2? Answer in one sentence.
  Model: 2 + 2 = 4.
  (1.1s, 9 tokens)

MULTI-TURN CONVERSATION
  User: What is the capital of France?
  Model: The capital of France is Paris.
  User: What is its population?
  Model: As of November 2023, the population of Paris is approximately 2.1 million people.
```

Resolves a GGUF model from sharded flat-repo, loads with `llama-cpp-python`, then runs
single-turn and multi-turn chat completion via the OpenAI-compatible API.

### Node.js + LLM — GGUF Chat with node-llama-cpp

```bash
npm install node-llama-cpp

node example-inference-llm.mjs \
  --source /path/to/pkg-gemma3 \
  --max-tokens 128
```

Same pipeline in Node.js — resolves GGUF shards, loads with `node-llama-cpp`,
runs `LlamaChatSession` for multi-turn conversation with automatic history management.

---

## Integration Guide

### Complete working example

See `example.html` for a self-contained, commented page that loads an embedding
model through the Service Worker, shows download progress, and runs inference.
Copy it alongside the packaged model directory and the required JS/WASM files,
start the dev server, and open in a browser.

### With Transformers.js (ONNX models)

```javascript
import { AutoModel, AutoTokenizer, env } from '@huggingface/transformers';

// Point at your local path (SW will intercept)
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.backends.onnx.wasm.wasmPaths = '/';  // serve ort-wasm files from root

// Register SW
await navigator.serviceWorker.register('/model-sw.js');
await navigator.serviceWorker.ready;

navigator.serviceWorker.controller.postMessage({
  type: 'MODEL_SW_INIT',
  sources: [{
    pathPrefix: '/models/my-model/',
    cdnBase: 'https://cdn.jsdelivr.net/gh/user/repo@tag',
    progress: true,        // optional
    manifest: 'q4f16',     // optional
  }]
});

// Load — Transformers.js fetches from /models/my-model/...
// SW intercepts, reassembles from CDN shards
const tokenizer = await AutoTokenizer.from_pretrained('/models/my-model/');
const model = await AutoModel.from_pretrained('/models/my-model/', {
  dtype: 'q4f16',
});

// Signal completion (for progress tracking)
navigator.serviceWorker.controller.postMessage({
  type: 'MODEL_SW_COMPLETE',
  pathPrefix: '/models/my-model/',
});
```

### With wllama (GGUF models)

```javascript
import { Wllama } from '@wllama/wllama';

// SW intercepts /models/llm/ → fetches shards from CDN, reassembles transparently
const wllama = new Wllama({
  'single-thread/wllama.wasm': '/wllama/single-thread/wllama.wasm',
  'multi-thread/wllama.wasm': '/wllama/multi-thread/wllama.wasm',
});

// IMPORTANT: use absolute URL — wllama's internal Worker needs a full origin
await wllama.loadModelFromUrl(`${location.origin}/models/llm/model-Q4_K_M.gguf`, {
  n_ctx: 2048,
  n_threads: 1,
});

const reply = await wllama.createChatCompletion(
  [{ role: 'user', content: 'Hello!' }],
  { nPredict: 128, sampling: { temp: 0.7 } }
);
```

### Without the Service Worker (direct CDN)

If you can't use a Service Worker (e.g., cross-origin restrictions), you can
read the filemap yourself and concatenate shards manually:

```javascript
const resp = await fetch('https://cdn.jsdelivr.net/gh/user/repo@tag/filemap.json');
const filemap = await resp.json();

const entry = filemap.files['onnx/model_q4f16.onnx_data'];
const parts = [];
for (const shard of entry.shards) {
  const r = await fetch(`https://cdn.jsdelivr.net/gh/user/repo@tag/${shard.file}`);
  parts.push(await r.arrayBuffer());
}
const fullFile = new Blob(parts);
```

---

## Progress Tracking

Progress tracking is **opt-in** — set `progress: true` in the source config.
When disabled (the default), the SW has zero overhead: no Maps, no timers,
no `postMessage` calls.

### Option A: SW progress only (byte-level accuracy)

```javascript
// Listen for SW progress messages
navigator.serviceWorker.addEventListener('message', (e) => {
  if (e.data?.type === 'MODEL_SW_PROGRESS') {
    const { percent, modelLoaded, modelTotal, done, manifest, mode } = e.data;
    updateProgressBar(percent);
    if (done) showComplete();
  }
});
```

The `MODEL_SW_PROGRESS` message fields:

| Field | Type | Description |
|-------|------|-------------|
| `pathPrefix` | string | Which source this progress is for |
| `file` | string | Last file that had progress |
| `fileLoaded` | number | Bytes loaded for that file |
| `fileTotal` | number | Total bytes for that file |
| `modelLoaded` | number | Total bytes loaded across all files |
| `modelTotal` | number | Denominator (from manifest or all files) |
| `percent` | number | 0–100, rounded integer |
| `done` | boolean | True when finalized |
| `manifest` | string | Currently selected manifest name (or null) |
| `mode` | string | `'explicit'`, `'adaptive'`, or `'fallback'` |

### Option B: Transformers.js progress_callback only (no SW progress)

You don't need SW progress tracking at all. Transformers.js has its own:

```javascript
const model = await AutoModel.from_pretrained('/models/my-model/', {
  dtype: 'q4f16',
  progress_callback: (info) => {
    switch (info.status) {
      case 'initiate':
        // File download starting: info.name, info.file
        break;
      case 'progress':
        // Download progress: info.file, info.progress (0-100), info.loaded, info.total
        break;
      case 'done':
        // Single file complete: info.file
        break;
      case 'ready':
        // ALL files loaded, model ready: info.task, info.model
        break;
    }
  },
});
```

**Limitation:** `progress_callback` reports per-file, not whole-model. You don't
know the total model size upfront, so you can't show a single accurate bar until
all files are initiated. The SW approach knows the total from the manifest before
any download starts.

### Option C: Both (dual-perspective, used in the test harness)

Use the SW for byte-accurate download progress and Transformers.js for lifecycle:

```
[SW: downloading 0% → 50% → 100%]  →  [TF.js: compiling...]  →  [Ready ✓]
```

The SW bar fills during download. Once bytes arrive, Transformers.js handles
ONNX session creation (CPU-bound, no network). The SW can't see compilation.

---

## Manifests & Multi-Quantization

### The problem

A merged filemap with fp32 + fp16 + q8 + q4f16 totals 2.3 GB. The page loads
q4f16 (200 MB). Without manifests, the progress bar denominator is 2.3 GB —
progress maxes at ~9%.

### The solution: manifests

Manifests are named subsets of the filemap's files, with pre-computed total sizes:

```json
{
  "manifests": {
    "q4f16": {
      "files": ["config.json", "tokenizer.json", "onnx/model_q4f16.onnx", "onnx/model_q4f16.onnx_data"],
      "size": 202305929
    },
    "fp16": {
      "files": ["config.json", "tokenizer.json", "onnx/model_fp16.onnx", "onnx/model_fp16.onnx_data"],
      "size": 614728800
    }
  }
}
```

Shared files (config, tokenizer) appear in every manifest. The `size` field is the
sum of all listed files' sizes.

### Three progress modes

**Explicit** — you know the variant at init time:
```javascript
{ pathPrefix: '/m/', cdnBase: '...', progress: true, manifest: 'q4f16' }
// Denominator: 202 MB. Fixed. Never changes.
```

**Adaptive** — SW figures it out by observing requests:
```javascript
{ pathPrefix: '/m/', cdnBase: '...', progress: true }
// no manifest specified
// 1. Starts with LARGEST manifest as denominator (pessimistic)
// 2. When model_q4f16.onnx is requested, narrows to manifests containing it
// 3. Switches denominator to matching manifest
// 4. Progress may jump forward, never backward (monotonic guarantee)
// 5. Idle-finalizes after 2s of no activity
```

**Fallback** — no manifests in filemap at all:
```javascript
// Old-style filemap without manifests section
// Denominator = all files. Works fine for single-quantization repos.
```

### Idle finalization

Manifests may include files the runtime never requests (README.md, generation_config.json).
Without finalization, progress stalls at 98%.

- **Explicit mode:** Send `MODEL_SW_COMPLETE` from the page after loading finishes.
- **Adaptive mode:** After all pending fetches complete and 2 seconds pass with no
  new requests, the SW shrinks the denominator to only actually-fetched files and
  emits `done: true`.

### GGUF metadata extraction

When packaging GGUF files, the packager uses `gguf-meta.py` to read each file's
header and extract:

- **Architecture** — `gemma3`, `llama`, `phi3`, `clip`, etc.
- **Classification** — `llm` (text model) or `mmproj` (vision/audio projector, detected
  by `mmproj` in the filename or architecture being `clip`, `mllama_vision`, etc.)
- **Quantization** — `Q4_0`, `Q4_K_M`, `F16`, `IQ4_XS`, etc. (from `general.file_type`
  metadata or filename pattern)
- **Model parameters** — context length, embedding dims, layer count, vocab size,
  head counts, etc.

This metadata is stored in `filemap.json` under the `gguf_metadata` key:

```json
{
  "gguf_metadata": {
    "model-q4_0.gguf": {
      "classification": "llm",
      "architecture": "gemma3",
      "quantization": "Q4_0",
      "context_length": 32768,
      "embedding_length": 1152,
      "block_count": 26,
      "vocab_size": 262144
    }
  }
}
```

The harness UI (or any client) can read this metadata to display model info, set
wllama parameters (`n_ctx`, etc.), or auto-select the right model variant.

### Multimodal manifest generation

When a packaged directory contains both LLM and mmproj GGUF files, the packager
automatically generates cross-permutation manifests:

```bash
# Input: model-q4_0.gguf (LLM), model-q8_0.gguf (LLM),
#        mmproj-f16.gguf (vision), mmproj-q8_0.gguf (vision)
./model-packager.sh -o ./pkg-multimodal /path/to/model/

# Generated manifests:
#   q4_0              ← LLM only (text generation)
#   q8_0              ← LLM only (higher quality)
#   mmproj_f16        ← vision projector only (rarely loaded alone)
#   mmproj_q8_0       ← vision projector only
#   q4_0+mmproj_f16   ← LLM + vision (small + fast)
#   q4_0+mmproj_q8_0  ← LLM + vision
#   q8_0+mmproj_f16   ← LLM + vision (big LLM, fast projector)
#   q8_0+mmproj_q8_0  ← LLM + vision (highest quality)
```

**Important:** Not all permutations may be valid. The packager prints a warning:

```
⚠ Generated 4 LLM+mmproj cross-permutation manifest(s) from 2 LLM × 2 mmproj.
  Some permutations may be invalid (architecture mismatch, incompatible quant).
  Review manifests in filemap.json and remove any invalid combinations.
```

Remove invalid manifests from `filemap.json` manually before deployment.

You can also use `gguf-meta.py` standalone to inspect any GGUF file:

```bash
python3 gguf-meta.py model.gguf                # Full JSON metadata
python3 gguf-meta.py --classify model.gguf      # Just: "llm" or "mmproj"
python3 gguf-meta.py --quant model.gguf          # Just: "Q4_0" etc.
python3 gguf-meta.py model.gguf mmproj.gguf      # Multiple files at once
```

---

## Test Harness

The test harness is a browser-based UI that loads two ONNX models (embedding + reranker)
and shows progress from both the Service Worker and Transformers.js perspectives.

### Directory structure

```
harness/
├── index.html              ← Main harness UI (Alpine.js + Transformers.js)
├── model-sw.js             ← Service Worker (copy from WebModelDelivery/)
├── serve.mjs               ← Dev server with Range support
├── run-test.mjs            ← Headless Chrome test runner
├── transformers.js         ← Transformers.js bundle (from npm)
├── alpine.min.js           ← Alpine.js (UI reactivity)
├── ort-wasm-simd-*.wasm    ← ONNX Runtime WASM files (from npm)
├── ort-wasm-simd-*.mjs     ← ONNX Runtime JS loaders
├── pkg-embedding/          ← Packaged embedding model (symlink or copy)
│   ├── filemap.json
│   └── *.shard.*
└── pkg-reranker/           ← Packaged reranker model
    ├── filemap.json
    └── *.shard.*
```

### Setting up the harness

**Quick setup (recommended):**
```bash
./setup-harness.sh
# Then package your models:
./model-packager.sh -o ./harness/pkg-embedding /path/to/onnx-embedding-model/
./model-packager.sh -o ./harness/pkg-reranker  /path/to/onnx-reranker-model/
```

**Manual setup:**

```bash
# 1. Create project directory
mkdir my-harness && cd my-harness

# 2. Install dependencies
npm init -y
npm install @huggingface/transformers onnxruntime-web

# 3. Copy runtime files into harness directory
# Transformers.js bundle:
cp node_modules/@huggingface/transformers/dist/transformers.js .

# ONNX Runtime WASM files (CRITICAL — must be served with correct MIME types):
cp node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm .
cp node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs .
cp node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm .
cp node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs .

# Alpine.js (optional, for the harness UI):
curl -o alpine.min.js https://cdn.jsdelivr.net/npm/alpinejs@3/dist/cdn.min.js

# 4. Copy model-sw.js
cp /path/to/WebModelDelivery/model-sw.js .

# 5. Package your models
/path/to/model-packager.sh -o ./pkg-embedding /path/to/onnx-model/
/path/to/model-packager.sh -o ./pkg-reranker /path/to/reranker-model/

# 6. Copy harness UI and test runner
cp /path/to/WebModelDelivery/harness-index.html index.html
cp /path/to/WebModelDelivery/run-test.mjs .
cp /path/to/WebModelDelivery/serve.mjs .

# 7. Start server
node serve.mjs
# → Server running at http://localhost:9XXX
```

---

## Test Harness (LLM / wllama)

The LLM harness loads a GGUF model (Gemma 3 1B) through wllama (llama.cpp compiled
to WebAssembly) and verifies the full pipeline: SW shard reassembly → GGUF loading →
chat completion.

### Directory structure

```
harness-llm/
├── index.html              ← LLM harness UI (vanilla JS + wllama)
├── model-sw.js             ← Service Worker (copy from WebModelDelivery/)
├── serve.mjs               ← Dev server with Range + COOP/COEP support
├── run-test-llm.mjs        ← Headless Chrome LLM test runner
├── wllama.js               ← wllama ESM bundle (from npm @wllama/wllama)
├── single-thread/
│   └── wllama.wasm         ← llama.cpp WASM (single-thread)
├── multi-thread/
│   └── wllama.wasm         ← llama.cpp WASM (multi-thread, needs COOP/COEP)
└── pkg-gemma3/             ← Packaged GGUF model (output of model-packager.sh)
    ├── filemap.json
    └── *.shard.*
```

### Setting up the LLM harness

**Quick setup (recommended):**
```bash
./setup-harness.sh --llm
# Then package your GGUF model:
./model-packager.sh -o ./harness-llm/pkg-gemma3 \
  --gguf-split /path/to/llama-gguf-split \
  /path/to/gemma-3-1b-it-qat-q4_0-gguf/
```

**Manual setup:**

```bash
# 1. Create harness directory
mkdir -p harness-llm/single-thread harness-llm/multi-thread && cd harness-llm

# 2. Install wllama
npm init -y && npm install @wllama/wllama

# 3. Copy wllama runtime files
cp node_modules/@wllama/wllama/esm/index.js wllama.js
cp node_modules/@wllama/wllama/esm/single-thread/wllama.wasm single-thread/
cp node_modules/@wllama/wllama/esm/multi-thread/wllama.wasm multi-thread/

# 4. Copy project files
cp /path/to/WebModelDelivery/model-sw.js .
cp /path/to/WebModelDelivery/serve.mjs .
cp /path/to/WebModelDelivery/run-test-llm.mjs .
cp /path/to/WebModelDelivery/harness-llm-index.html index.html

# 5. Package a GGUF model (requires llama-gguf-split in PATH)
/path/to/model-packager.sh -o ./pkg-gemma3 \
  --gguf-split /path/to/llama-gguf-split \
  /path/to/gemma-3-1b-it-qat-q4_0-gguf/

# 6. Start server
node serve.mjs
# → Open http://localhost:PORT in browser
```

### Key differences from ONNX harness

| Aspect | ONNX (Transformers.js) | GGUF (wllama) |
|--------|------------------------|---------------|
| Runtime | Transformers.js + ONNX Runtime WASM | wllama (llama.cpp WASM) |
| Model format | `.onnx` + `.onnx_data` | `.gguf` |
| Inference | Embedding, reranking, classification | Text generation (chat, completion) |
| Threading | Single-thread only | Single or multi-thread |
| URL format | Relative paths OK | **Must use absolute URLs** (`location.origin + path`) |
| COOP/COEP | Not required | Required for multi-thread mode |
| Typical model size | 90–600 MB | 500 MB – 8 GB |

### wllama integration notes

**Absolute URLs required.** wllama spawns an internal Web Worker that
cannot resolve root-relative paths (`/models/...`). Always use:
```javascript
await wllama.loadModelFromUrl(`${location.origin}/models/llm/model.gguf`, config);
// NOT: await wllama.loadModelFromUrl('/models/llm/model.gguf', config);
```

**SharedArrayBuffer (multi-thread).** For wllama's multi-thread mode, the server
must send COOP/COEP headers. The provided `serve.mjs` includes them:
```
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Opener-Policy: same-origin
```
Without these headers, wllama falls back to single-thread mode automatically.

**GGUF splitting for very large models.** Models over 2 GB must be pre-split with
`llama-gguf-split` (the packager handles this automatically). wllama loads split
GGUF shards as an array of URLs:
```javascript
await wllama.loadModelFromUrl([
  `${location.origin}/models/llm/model-00001-of-00003.gguf`,
  `${location.origin}/models/llm/model-00002-of-00003.gguf`,
  `${location.origin}/models/llm/model-00003-of-00003.gguf`,
], config);
```
Each split shard is independently CDN-chunked by the packager and reassembled by the SW.

### Running the LLM test (headless)

```bash
npm install puppeteer-core

# Full test (load + multi-turn inference + unload):
npm run test:llm

# Quick test (load only, skip slow inference):
npm run test:llm:quick

# Or run directly:
CHROME_PATH=/opt/google/chrome/chrome node run-test-llm.mjs --root ./harness-llm

# Quick test (load only, skip slow inference):
node run-test-llm.mjs --root ./harness-llm --skip-inference
```

The test validates:
- Service Worker registration and shard reassembly
- Progress tracking (monotonic, manifest-aware, finalization)
- COOP/COEP headers (required for multi-thread wllama)
- Model metadata (context length, layer count, vocab size)
- Single-turn inference with answer validation
- Multi-turn chat history preservation
- Model unload lifecycle

### Expected LLM test output

```
=== LLM Model Load (explicit manifest: q4_0) ===
  ✓ Model loaded in 89.0s

=== LLM (explicit:q4_0) ===
  SW msgs:  102
  Pcts:     2→4→6→8→10→...→96→100
  Monotonic: YES ✓ | Mode: explicit | Manifest: q4_0
  Done: true | Total: 957.1 MB

=== COOP/COEP & Threading ===
  crossOriginIsolated: true
  SharedArrayBuffer:   true
  ✓ COOP/COEP headers active — multi-thread wllama available

=== Model Metadata ===
  Info: Loaded in 89049ms · ctx: 2048 · params: 26L × 1152d · vocab: 262144
  ✓ Metadata contains ctx, params, vocab

=== LLM Inference — Turn 1 ===
  ✓ Response (6913ms): "Four"
  ✓ Response contains expected answer

=== LLM Inference — Turn 2 (multi-turn) ===
  ✓ Response (12024ms): "12"
  ✓ Multi-turn generation succeeded
  ✓ Chat history maintained across turns

=== Unload / Reload ===
  ✓ Model unloaded successfully

✓ All tests passed.
```

---

## Running Tests (Headless)

### Prerequisites

```bash
npm install puppeteer-core
# Ensure Chrome is installed (see "Finding Chrome" section above)
```

### Running the ONNX test

```bash
# Via npm scripts (from project root):
npm run test:onnx            # Full test (load + inference)
npm run test:onnx:quick      # Quick test (load only, skip inference)

# Or run directly:
export CHROME_PATH=/opt/google/chrome/chrome
node run-test.mjs --root ./harness
node run-test.mjs --root ./harness --skip-inference   # quick
```

### What the ONNX test does

1. Starts a local HTTP server with COOP/COEP headers (random port)
2. Launches headless Chrome with Puppeteer
3. Navigates to the harness page
4. Waits for Service Worker registration
5. Hooks into `MODEL_SW_PROGRESS` messages
6. Clicks "Load Model" for the embedding model
7. Waits for `ready ✓` in the page
8. Verifies: message count, monotonic progress, correct manifest, mode, done flag
9. Repeats for the reranker model (adaptive mode)
10. Runs embedding + reranker inference, verifies outputs
11. Takes screenshots at each stage
12. Exits with code 0 (pass) or 1 (fail)

### Expected ONNX output

```
=== EMBEDDING (explicit:q4f16) ===
  SW msgs:   15
  Pcts:      1→10→11→21→31→41→50→60→70→80→90→98→100
  Monotonic: YES ✓
  Mode:      explicit
  Manifest:  q4f16
  Done:      true
  Total:     188.4 MB

=== RERANKER (adaptive:quantized) ===
  SW msgs:   9
  Pcts:      0→9→29→49→70→90→97→100
  Monotonic: YES ✓
  Mode:      adaptive
  Manifest:  quantized
  Done:      true
  Total:     91.5 MB

=== Inference ===
  Embedding: 4d vector in 722ms → [5.5664, 1.1123, -4.3594, 5.0430, 2.0488, ...]
  ✓ Embedding inference produced vector output
  Reranker results:
    #1: Paris is the capital and largest city of France.
    #2: The Eiffel Tower is in Paris.
    #3: France is known for its cuisine and wine.
    #4: Berlin is the capital of Germany.
  ✓ Reranker produced ranked results

✓ All tests passed.
```

---

## Running Interactively (Manual Testing)

### Starting the server

```bash
# ONNX harness — from the project root:
node serve.mjs ./harness
# or via npm:
npm run serve:onnx

# LLM harness — from the project root:
node serve.mjs ./harness-llm
# or via npm:
npm run serve:llm

# From inside a harness directory:
cd harness && node serve.mjs
```

The server prints the full URL to open:

```
WebModelDelivery dev server
  Root:  /path/to/harness
  URL:   http://localhost:42371
  Open:  http://localhost:42371/index.html
  Range: ✓  MIME: ✓  CORS: ✓  SW-Allowed: ✓  COOP/COEP: ✓

Press Ctrl+C to stop.
```

**Port behavior:**
- Default: OS assigns a random free port (avoids conflicts in shared environments)
- Fixed port: `PORT=8080 node serve.mjs` (auto-retries nearby ports if busy)
- The server shuts down cleanly on Ctrl+C (SIGINT/SIGTERM)

### ONNX harness walkthrough

Open the URL printed by the server. The harness loads two models and shows dual progress:

1. **Service Worker registration** — the status badge shows "Active" (green) when ready
2. **Click "Load Model"** for Embedding (explicit manifest: q4f16, ~193 MB)
   - **Left panel:** Transformers.js runtime progress (per-file initiate → download → done → ready)
   - **Right panel:** SW transport progress (total bytes via shard reassembly)
3. **Click "Load Model"** for Reranker (adaptive manifest, ~94 MB)
   - Observe how adaptive mode starts with a pessimistic denominator and narrows
4. **Run inference:**
   - Type text in the Embedding input and click "Embed" — see vector dimensions + values
   - Click "Rerank sample docs" — see ranked relevance scores for a canned query
5. **Event Log** (bottom) — shows timestamped events from both runtime and transport layers
6. **"Clear cache"** — purges the SW shard cache (Cache API), forcing re-download on next load

### LLM harness walkthrough

1. **Click "Load Model"** — downloads Gemma 3 1B Q4_0 (~957 MB) through SW shard reassembly
   - Progress bar shows SW transport progress (bytes/total, manifest, mode)
   - Status line shows wllama download progress
2. **Chat interface** — type a message and press Enter or click Send
   - Tests multi-turn conversation (chat history is preserved across turns)
   - Model metadata shown after load: layers × embedding dims, vocab size, context length
3. **Click "Unload"** — releases WASM memory, resets the interface
4. **Event Log** — shows wllama lifecycle events, SW messages, and inference timing

### Python server (fallback)

```bash
# Python's built-in server does NOT support Range requests.
# ONNX models won't load. Use only for testing basic SW registration.
python3 -m http.server 8080

# For Range support, use this one-liner:
python3 -c "
import http.server, os, re

class RangeHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.mjs': 'text/javascript',
        '.wasm': 'application/wasm',
        '.onnx': 'application/octet-stream',
    }
    def do_GET(self):
        rh = self.headers.get('Range')
        if not rh:
            return super().do_GET()
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404); return
        m = re.match(r'bytes=(\d+)-(\d*)', rh)
        if not m:
            return super().do_GET()
        size = os.path.getsize(path)
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
        length = end - start + 1
        self.send_response(206)
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', length)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Service-Worker-Allowed', '/')
        self.end_headers()
        with open(path, 'rb') as f:
            f.seek(start)
            self.wfile.write(f.read(length))

http.server.HTTPServer(('', 8080), RangeHandler).serve_forever()
" 
```

### Server requirements checklist

Your server MUST support these for the system to work:

| Requirement | Why | What breaks without it |
|-------------|-----|----------------------|
| `.mjs` → `text/javascript` | ONNX Runtime loads `.mjs` modules | "Failed to fetch dynamically imported module" |
| `.wasm` → `application/wasm` | WebAssembly requires correct MIME | WASM compilation fails |
| Range requests (206) | ONNX Runtime reads model files in chunks | Model loading hangs or fails |
| `Service-Worker-Allowed: /` | Allows SW scope to cover the whole origin | SW registration fails |
| CORS (`Access-Control-Allow-Origin`) | Only needed if SW and CDN are cross-origin | Fetch errors from SW |
| COOP/COEP headers | Required for wllama multi-thread mode | Falls back to single-thread (slower) |


## Filemap Format Reference

```jsonc
{
  "version": 5,
  "generator": "model-packager v4.4.0",
  "created": "2025-02-22T12:00:00Z",

  // Files section: virtual path → metadata
  "files": {
    "config.json": {
      "size": 1234,                          // Original file size in bytes
      "sha256": "abc123...",                 // SHA256 of original file
      "cdn_file": "config.json"             // CDN filename (if no shards)
    },
    "onnx/model_q4f16.onnx_data": {
      "size": 175234567,
      "sha256": "def456...",
      "shards": [                            // Present when file was split
        {
          "file": "model_q4f16.onnx_data.shard.000",  // CDN filename
          "offset": 0,                                  // Byte offset in original
          "size": 19922944                              // Shard size
        },
        {
          "file": "model_q4f16.onnx_data.shard.001",
          "offset": 19922944,
          "size": 19922944
        }
        // ...
      ]
    },
    "model-Q4_K_M.gguf": {
      "size": 4500000000,
      "sha256": "...",
      "gguf_source": "original-model-Q4_K_M.gguf",    // GGUF only: original filename
      "shards": [ ... ]
    }
  },

  // Manifests section: named subsets for progress tracking
  "manifests": {
    "q4f16": {
      "files": ["config.json", "tokenizer.json", "onnx/model_q4f16.onnx", "onnx/model_q4f16.onnx_data"],
      "size": 202305929                      // Pre-computed sum of listed files
    },
    "fp16": {
      "files": ["config.json", "tokenizer.json", "onnx/model_fp16.onnx", "onnx/model_fp16.onnx_data"],
      "size": 614728800
    }
  }
}
```

---

## Files in This Package

| File | Purpose |
|------|---------|
| **Core** | |
| `model-packager.sh` | Shell script to package model files into CDN-safe shards |
| `gguf-meta.py` | GGUF metadata extractor — used by packager for auto-classification |
| `model-sw.js` | Service Worker that intercepts fetches and reassembles shards (browser) |
| `model-resolver-node.mjs` | Node.js resolver — resolve/fetch-hook for local flat-repos and CDN |
| `model_resolver.py` | Python resolver — resolve models from local flat-repos or CDN |
| **Downloaders** | |
| `model-downloader.py` | Python script to download/reassemble model files from a filemap |
| `model-downloader.mjs` | Node.js script to download/reassemble model files from a filemap |
| **Examples** | |
| `example-inference-onnx.py` | Python end-to-end: resolve → tokenize → embedding + reranker inference |
| `example-inference-onnx.mjs` | Node.js end-to-end: resolve → tokenize → embedding + reranker inference |
| `example-inference-llm.py` | Python end-to-end: resolve GGUF → llama-cpp-python multi-turn chat |
| `example-inference-llm.mjs` | Node.js end-to-end: resolve GGUF → node-llama-cpp multi-turn chat |
| `example.html` | Minimal browser ONNX integration example |
| **Test Harness** | |
| `serve.mjs` | Dev server with Range request support, COOP/COEP headers |
| `run-test.mjs` | Headless Chrome end-to-end test runner (ONNX models) |
| `run-test-llm.mjs` | Headless Chrome end-to-end test runner (GGUF / wllama) |
| `setup-harness.sh` | One-command harness setup — supports `--llm` for GGUF/wllama mode |
| `harness-index.html` | Full test harness UI for ONNX models (embedding + reranker) |
| `harness-llm-index.html` | Full test harness UI for GGUF / wllama LLM chat |
| `test-resolvers.py` | Comprehensive Python resolver test suite (14 tests) |
| `test-resolvers.mjs` | Comprehensive Node.js resolver test suite (10 tests) |
| `README.md` | This documentation |

---

## Troubleshooting

### "Failed to fetch dynamically imported module: ...ort-wasm-simd-threaded.jsep.mjs"

Your server isn't serving `.mjs` files with `text/javascript` MIME type. Use the
provided `serve.mjs` or add the MIME mapping to your server config.

### Progress stuck at 98%

Manifest includes files the runtime doesn't fetch (README.md, etc.).
Send `MODEL_SW_COMPLETE` after `from_pretrained()` resolves:
```javascript
navigator.serviceWorker.controller.postMessage({
  type: 'MODEL_SW_COMPLETE', pathPrefix: '/models/my-model/'
});
```

### Progress shows ~9% and stops

Multi-quantization filemap without manifests. Either:
- Add `manifest: 'q4f16'` to your source config, or
- Re-package with `model-packager.sh` v4.3+ (auto-generates manifests)

### Service Worker not intercepting requests

- Check the SW is registered: `navigator.serviceWorker.controller` should not be null
- First load may need a page reload (SW activates after install)
- `pathPrefix` must exactly match the URL path your framework requests
- The server must send `Service-Worker-Allowed: /` header

### ONNX model loading hangs

Server likely doesn't support HTTP Range requests. ONNX Runtime reads models
in chunks via Range headers. Use the provided `serve.mjs` or ensure your
server returns proper 206 responses.

### Cache not clearing

The SW caches shards in `model-shards-v1`. Send `MODEL_SW_CLEAR_CACHE` or:
```javascript
await caches.delete('model-shards-v1');
```
Then reload the page. In Chrome DevTools: Application → Cache Storage → delete.

### wllama: "Failed to parse URL from /models/..."

wllama's internal Web Worker cannot resolve root-relative paths. Use absolute URLs:
```javascript
// ✗ Broken — Worker can't resolve this
await wllama.loadModelFromUrl('/models/llm/model.gguf');

// ✓ Works — Worker sees the full origin
await wllama.loadModelFromUrl(`${location.origin}/models/llm/model.gguf`);
```

### wllama: slow inference (single-thread fallback)

If wllama is using single-thread mode when you expect multi-thread, your server
likely isn't sending COOP/COEP headers. Add these to all responses:
```
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Opener-Policy: same-origin
```
The provided `serve.mjs` includes them. Check with:
`self.crossOriginIsolated` → should be `true` in the browser console.

### wllama: "munmap failed: Invalid argument"

This is a harmless warning from llama.cpp's WASM build. The `munmap` syscall
is not fully supported in Emscripten's virtual filesystem. It does not affect
model loading or inference.

### wllama: GGUF file >2 GB

wllama requires GGUF files under 2 GB each. The packager automatically
splits large GGUFs with `llama-gguf-split`. Pass all split shards as an
array of URLs to `loadModelFromUrl()`. The packager's filemap includes
`gguf_source` metadata to group split shards for manifest generation.
