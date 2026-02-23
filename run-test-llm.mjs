#!/usr/bin/env node
/**
 * run-test-llm.mjs — Headless end-to-end test for wllama / GGUF model delivery
 *
 * Starts a local server, launches headless Chrome, loads a GGUF model through the
 * Service Worker, verifies progress tracking, runs a chat completion, and takes
 * screenshots.
 *
 * Prerequisites:
 *   npm install puppeteer-core @wllama/wllama
 *   Chrome or Chromium installed (see CHROME_PATH below)
 *   A packaged GGUF model in pkg-gemma3/ (via model-packager.sh)
 *
 * Usage:
 *   node run-test-llm.mjs                                # Auto-detect Chrome
 *   CHROME_PATH=/usr/bin/chromium node run-test-llm.mjs  # Explicit Chrome path
 *   node run-test-llm.mjs --root ./my-harness            # Custom harness directory
 *   node run-test-llm.mjs --screenshots ./shots           # Custom screenshot dir
 *   node run-test-llm.mjs --skip-inference                # Skip slow inference step
 *
 * Environment variables:
 *   CHROME_PATH    Path to Chrome/Chromium binary
 *   PORT           Fixed server port (default: random)
 *   TIMEOUT        Model load timeout in ms (default: 300000 = 5 min)
 */

import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// ─── Configuration ──────────────────────────────────────────────────────────

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const args = process.argv.slice(2);
const getArg = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const hasFlag = (name) => args.includes(name);

const ROOT = path.resolve(getArg('--root', '.'));
const SCREENSHOT_DIR = path.resolve(getArg('--screenshots', '/tmp'));
const TIMEOUT = parseInt(process.env.TIMEOUT || '300000', 10);
const SKIP_INFERENCE = hasFlag('--skip-inference');

// Chrome path discovery
const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  '/opt/google/chrome/chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium-browser',
  '/usr/bin/chromium',
  '/snap/bin/chromium',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
].filter(Boolean);

function findChrome() {
  for (const p of CHROME_CANDIDATES) {
    try { if (fs.existsSync(p)) return p; } catch {}
  }
  return null;
}

// ─── MIME types ─────────────────────────────────────────────────────────────

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.wasm': 'application/wasm', '.css': 'text/css',
  '.onnx': 'application/octet-stream', '.model': 'application/octet-stream',
  '.gguf': 'application/octet-stream',
  '.txt': 'text/plain', '.md': 'text/markdown',
};

// ─── HTTP Server (with Range support) ───────────────────────────────────────

function startServer() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const url = new URL(req.url, `http://localhost`);
      let fp = path.join(ROOT, decodeURIComponent(url.pathname));
      if (url.pathname === '/') fp = path.join(ROOT, 'index.html');
      try { fp = fs.realpathSync(fp); } catch {
        res.writeHead(404); res.end('Not found'); return;
      }
      if (!fs.existsSync(fp) || fs.statSync(fp).isDirectory()) {
        res.writeHead(404); res.end('Not found'); return;
      }
      const ext = path.extname(fp).toLowerCase();
      const mime = MIME[ext] || 'application/octet-stream';
      const stat = fs.statSync(fp);

      // Range requests
      const rh = req.headers.range;
      if (rh) {
        const m = rh.match(/bytes=(\d+)-(\d*)/);
        if (m) {
          const s = parseInt(m[1], 10), e = m[2] ? parseInt(m[2], 10) : stat.size - 1;
          if (s > e || s >= stat.size) {
            res.writeHead(416, { 'Content-Range': `bytes */${stat.size}` });
            res.end(); return;
          }
          res.writeHead(206, {
            'Content-Range': `bytes ${s}-${e}/${stat.size}`,
            'Accept-Ranges': 'bytes', 'Content-Length': e - s + 1,
            'Content-Type': mime, 'Access-Control-Allow-Origin': '*',
            'Service-Worker-Allowed': '/',
            'Cross-Origin-Embedder-Policy': 'require-corp',
            'Cross-Origin-Opener-Policy': 'same-origin',
          });
          fs.createReadStream(fp, { start: s, end: e }).pipe(res);
          return;
        }
      }

      res.writeHead(200, {
        'Content-Type': mime, 'Content-Length': stat.size,
        'Accept-Ranges': 'bytes', 'Access-Control-Allow-Origin': '*',
        'Service-Worker-Allowed': '/',
        'Cross-Origin-Embedder-Policy': 'require-corp',
        'Cross-Origin-Opener-Policy': 'same-origin',
      });
      fs.createReadStream(fp).pipe(res);
    });

    const port = parseInt(process.env.PORT || '0', 10);
    server.listen(port, () => resolve({ server, port: server.address().port }));
  });
}

// ─── Test runner ────────────────────────────────────────────────────────────

async function main() {
  console.log('WebModelDelivery — LLM Headless Test Runner\n');

  // Find Chrome
  const chromePath = findChrome();
  if (!chromePath) {
    console.error('ERROR: Chrome/Chromium not found. Set CHROME_PATH or install Chrome.');
    console.error('Searched:', CHROME_CANDIDATES.join(', '));
    process.exit(1);
  }
  console.log(`Chrome:      ${chromePath}`);
  console.log(`Harness:     ${ROOT}`);
  console.log(`Screenshots: ${SCREENSHOT_DIR}`);
  console.log(`Timeout:     ${TIMEOUT}ms`);
  console.log(`Inference:   ${SKIP_INFERENCE ? 'SKIP' : 'enabled'}\n`);

  // Verify harness files exist
  const required = ['index.html', 'model-sw.js', 'wllama.js', 'single-thread/wllama.wasm'];
  for (const f of required) {
    if (!fs.existsSync(path.join(ROOT, f))) {
      console.error(`ERROR: Missing ${f} in ${ROOT}`);
      process.exit(1);
    }
  }

  // Verify packaged model exists
  if (!fs.existsSync(path.join(ROOT, 'pkg-gemma3', 'filemap.json'))) {
    console.error('ERROR: Missing pkg-gemma3/filemap.json — run model-packager.sh first');
    process.exit(1);
  }

  // Start server
  const { server, port } = await startServer();
  console.log(`Server:      http://localhost:${port}\n`);

  // Launch browser
  const puppeteer = await import('puppeteer-core');
  const browser = await puppeteer.default.launch({
    headless: 'new',
    executablePath: chromePath,
    args: [
      '--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu',
      '--disable-dev-shm-usage',
      // Allow SharedArrayBuffer for multi-thread WASM
      '--enable-features=SharedArrayBuffer',
    ],
  });

  // Cleanup handler — ensures browser + server close on crash or signal
  const cleanup = async (code = 1) => {
    try { await browser.close(); } catch {}
    try { server.close(); } catch {}
    process.exit(code);
  };
  process.on('SIGINT', () => cleanup(130));
  process.on('SIGTERM', () => cleanup(143));
  process.on('unhandledRejection', (err) => {
    console.error('UNHANDLED:', err?.message || err);
    cleanup(1);
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 1000 });

  // Collect SW progress messages
  const swLog = [];
  await page.exposeFunction('_swProgressHook', (json) => swLog.push(JSON.parse(json)));

  // Console logging
  page.on('console', (msg) => {
    const text = msg.text();
    if (text.includes('model-sw')) console.log(`  [SW] ${text}`);
    else if (text.includes('wllama') || text.includes('llama')) console.log(`  [WLLAMA] ${text.slice(0, 150)}`);
  });
  page.on('pageerror', (e) => console.log(`  [PAGE ERROR] ${e.message.slice(0, 300)}`));

  let allPassed = true;

  // ─── Navigate and wait for SW ─────────────────────────────────────────

  console.log('=== Page Load ===');
  await page.goto(`http://localhost:${port}`, { waitUntil: 'networkidle0', timeout: 30000 });
  await sleep(5000);

  // Handle first-install reload
  const hasController = await page.evaluate(() => !!navigator.serviceWorker.controller);
  if (!hasController) {
    console.log('  SW installed, waiting for controller...');
    await sleep(3000);
    const still = await page.evaluate(() => !!navigator.serviceWorker.controller);
    if (!still) {
      console.log('  Reloading for SW controller...');
      await page.reload({ waitUntil: 'networkidle0', timeout: 20000 });
      await sleep(3000);
    }
  }

  // Hook SW progress listener
  await page.evaluate(() => {
    navigator.serviceWorker.addEventListener('message', (e) => {
      if (e.data?.type === 'MODEL_SW_PROGRESS') window._swProgressHook(JSON.stringify(e.data));
    });
  });

  const swState = await page.evaluate(() => navigator.serviceWorker.controller?.state || 'none');
  console.log(`  SW state: ${swState}`);
  await screenshot(page, 'llm_01_init');

  // ─── Load Model ───────────────────────────────────────────────────────

  console.log('\n=== LLM Model Load (explicit manifest: q4_0) ===');
  swLog.length = 0;

  // Click "Load Model"
  await page.evaluate(() => {
    const btn = document.getElementById('btn-load');
    if (btn && !btn.disabled) btn.click();
  });

  console.log('  Waiting for model load (this may take several minutes)...');

  try {
    await page.waitForFunction(
      () => window._modelLoaded === true || window._modelError,
      { timeout: TIMEOUT }
    );

    const error = await page.evaluate(() => window._modelError);
    if (error) {
      console.log(`  ✗ Model load ERROR: ${error}`);
      allPassed = false;
    } else {
      const loadTime = await page.evaluate(() => window._loadTimeMs);
      console.log(`  ✓ Model loaded in ${(loadTime / 1000).toFixed(1)}s`);
    }
  } catch {
    console.log(`  ✗ TIMEOUT — model did not load within ${TIMEOUT}ms`);
    allPassed = false;
  }

  await sleep(3000); // Let idle finalization complete
  await screenshot(page, 'llm_02_model_loaded');

  // Report SW progress
  const llmMsgs = swLog.filter(p => p.pathPrefix?.includes('llm'));
  reportProgress('LLM', llmMsgs);

  if (llmMsgs.length > 0) {
    const last = llmMsgs[llmMsgs.length - 1];
    if (!last.done) {
      console.log('  ⚠ SW progress did not finalize (done: false)');
    }
    // Verify monotonic
    let monotonic = true;
    const pcts = llmMsgs.map(p => p.percent);
    for (let i = 1; i < pcts.length; i++) {
      if (pcts[i] < pcts[i - 1]) { monotonic = false; break; }
    }
    if (!monotonic) {
      console.log('  ✗ Progress was NOT monotonic');
      allPassed = false;
    }
  }

  // ─── COOP/COEP & Threading check ──────────────────────────────────────
  console.log('\n=== COOP/COEP & Threading ===');
  const coopCheck = await page.evaluate(() => {
    return {
      crossOriginIsolated: self.crossOriginIsolated,
      hasSharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    };
  });
  console.log(`  crossOriginIsolated: ${coopCheck.crossOriginIsolated}`);
  console.log(`  SharedArrayBuffer:   ${coopCheck.hasSharedArrayBuffer}`);
  if (coopCheck.crossOriginIsolated) {
    console.log('  ✓ COOP/COEP headers active — multi-thread wllama available');
  } else {
    console.log('  ⚠ COOP/COEP not active — wllama will use single-thread only');
  }

  // ─── Model Metadata ──────────────────────────────────────────────────
  console.log('\n=== Model Metadata ===');
  const metadata = await page.evaluate(() => {
    const info = document.getElementById('model-info')?.textContent || '';
    return {
      infoText: info,
      loaded: window._modelLoaded,
      loadTimeMs: window._loadTimeMs,
    };
  });

  if (metadata.loaded) {
    console.log(`  Info: ${metadata.infoText}`);
    console.log(`  Load time: ${(metadata.loadTimeMs / 1000).toFixed(1)}s`);

    // Validate metadata contains expected fields
    const hasCtx = metadata.infoText.includes('ctx:');
    const hasParams = metadata.infoText.includes('params:');
    const hasVocab = metadata.infoText.includes('vocab:');
    if (hasCtx && hasParams && hasVocab) {
      console.log('  ✓ Metadata contains ctx, params, vocab');
    } else {
      console.log(`  ✗ Missing metadata fields (ctx:${hasCtx} params:${hasParams} vocab:${hasVocab})`);
      allPassed = false;
    }
  }

  // ─── Inference (Turn 1) ──────────────────────────────────────────────
  if (!SKIP_INFERENCE) {
    console.log('\n=== LLM Inference — Turn 1 ===');

    const modelLoaded = await page.evaluate(() => window._modelLoaded);
    if (!modelLoaded) {
      console.log('  Skipping inference — model not loaded');
    } else {
      // Type a message and send
      await page.type('#chat-input', 'What is 2+2? Answer in one word.');
      await page.click('#btn-send');

      console.log('  Waiting for generation...');

      try {
        await page.waitForFunction(
          () => window._lastResponse && window._lastResponse.length > 0,
          { timeout: 120000 }
        );

        const response = await page.evaluate(() => window._lastResponse);
        const responseMs = await page.evaluate(() => window._lastResponseMs);
        console.log(`  ✓ Response (${responseMs}ms): "${response.slice(0, 100).replace(/\n/g, '\\n')}${response.length > 100 ? '...' : ''}"`);
        console.log(`  Response length: ${response.length} chars`);

        if (response.length < 1) {
          console.log('  ✗ Response was empty');
          allPassed = false;
        }

        // Validate response contains something related to "4" or "four"
        const lower = response.toLowerCase().trim();
        if (lower.includes('four') || lower.includes('4')) {
          console.log('  ✓ Response contains expected answer');
        } else {
          console.log(`  ⚠ Response may not contain expected answer (got: "${lower}")`);
        }
      } catch {
        console.log('  ✗ TIMEOUT — inference did not complete within 120s');
        allPassed = false;
      }

      await screenshot(page, 'llm_03_inference_t1');

      // ─── Inference (Turn 2 — multi-turn) ───────────────────────────────
      console.log('\n=== LLM Inference — Turn 2 (multi-turn) ===');

      await page.evaluate(() => { window._lastResponse = ''; window._lastResponseMs = 0; });
      await page.type('#chat-input', 'Now multiply that by 3.');
      await page.click('#btn-send');

      console.log('  Waiting for generation...');

      try {
        await page.waitForFunction(
          () => window._lastResponse && window._lastResponse.length > 0,
          { timeout: 120000 }
        );

        const response2 = await page.evaluate(() => window._lastResponse);
        const response2Ms = await page.evaluate(() => window._lastResponseMs);
        console.log(`  ✓ Response (${response2Ms}ms): "${response2.slice(0, 100).replace(/\n/g, '\\n')}${response2.length > 100 ? '...' : ''}"`);

        if (response2.length < 1) {
          console.log('  ✗ Multi-turn response was empty');
          allPassed = false;
        } else {
          console.log('  ✓ Multi-turn generation succeeded');
        }

        // Check chat history length
        const histLen = await page.evaluate(() =>
          document.querySelectorAll('.chat-msg').length
        );
        console.log(`  Chat messages in DOM: ${histLen}`);
        if (histLen >= 5) { // system + user1 + assistant1 + user2 + assistant2
          console.log('  ✓ Chat history maintained across turns');
        } else {
          console.log(`  ✗ Expected ≥5 chat messages, got ${histLen}`);
          allPassed = false;
        }
      } catch {
        console.log('  ✗ TIMEOUT — multi-turn inference did not complete within 120s');
        allPassed = false;
      }

      await screenshot(page, 'llm_04_inference_t2');
    }
  } else {
    console.log('\n=== Inference skipped (--skip-inference) ===');
  }

  // ─── Unload / Reload test ────────────────────────────────────────────
  console.log('\n=== Unload / Reload ===');
  const wasLoaded = await page.evaluate(() => window._modelLoaded);
  if (wasLoaded) {
    await page.evaluate(() => window.unloadModel());
    await sleep(1000);
    const afterUnload = await page.evaluate(() => window._modelLoaded);
    if (!afterUnload) {
      console.log('  ✓ Model unloaded successfully');
    } else {
      console.log('  ✗ Model still reports loaded after unload');
      allPassed = false;
    }
  } else {
    console.log('  Skipping unload — model was not loaded');
  }

  // ─── Event Log ────────────────────────────────────────────────────────

  console.log('\n=== Event Log (last 20) ===');
  const log = await page.evaluate(() =>
    [...document.querySelectorAll('.log-entry')].slice(-20).map(e => e.textContent)
  );
  log.forEach(l => console.log(`  ${l}`));

  await screenshot(page, 'llm_05_final');

  // ─── Cleanup ──────────────────────────────────────────────────────────

  await browser.close();
  server.close();

  console.log('\n=== Test complete ===');
  if (allPassed) {
    console.log('✓ All tests passed.');
  } else {
    console.log('✗ Some tests failed.');
    process.exit(1);
  }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function screenshot(page, name) {
  const fp = path.join(SCREENSHOT_DIR, `harness_${name}.png`);
  await page.screenshot({ path: fp, fullPage: true });
  console.log(`  📸 ${fp}`);
}

function reportProgress(label, msgs) {
  if (msgs.length === 0) {
    console.log(`  No SW progress messages received`);
    return;
  }

  const pcts = msgs.map(p => p.percent);
  const last = msgs[msgs.length - 1];

  let monotonic = true;
  for (let i = 1; i < pcts.length; i++) {
    if (pcts[i] < pcts[i - 1]) { monotonic = false; break; }
  }

  // Deduplicate consecutive identical percentages for display
  const uniquePcts = [pcts[0]];
  for (let i = 1; i < pcts.length; i++) {
    if (pcts[i] !== pcts[i - 1]) uniquePcts.push(pcts[i]);
  }

  console.log(`\n=== ${label.toUpperCase()} (explicit:q4_0) ===`);
  console.log(`  SW msgs:  ${msgs.length}`);
  console.log(`  Pcts:     ${uniquePcts.join('→')}`);
  console.log(`  Monotonic: ${monotonic ? 'YES ✓' : 'NO ✗'}`);
  console.log(`  Mode:      ${last.mode}`);
  console.log(`  Manifest:  ${last.manifest || '(none)'}`);
  console.log(`  Done:      ${last.done}`);
  console.log(`  Total:     ${(last.modelTotal / 1048576).toFixed(1)} MB`);
}

// ─── Run ────────────────────────────────────────────────────────────────────

main().catch(err => {
  console.error('FATAL:', err.message);
  process.exit(1);
});
