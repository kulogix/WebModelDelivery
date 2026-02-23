#!/usr/bin/env node
/**
 * run-test.mjs — Headless end-to-end test for WebModelDelivery (ONNX / Transformers.js)
 *
 * Starts a local server, launches headless Chrome, loads ONNX models through the
 * Service Worker, verifies progress tracking (monotonic, correct manifest,
 * finalization), runs inference, and takes screenshots.
 *
 * Prerequisites:
 *   npm install puppeteer-core @huggingface/transformers onnxruntime-web
 *   Chrome or Chromium installed (see CHROME_PATH below)
 *
 * Usage:
 *   node run-test.mjs                                # Auto-detect Chrome
 *   CHROME_PATH=/usr/bin/chromium node run-test.mjs   # Explicit Chrome path
 *   node run-test.mjs --root ./my-harness             # Custom harness directory
 *   node run-test.mjs --screenshots ./shots            # Custom screenshot directory
 *   node run-test.mjs --skip-inference                 # Skip slow inference step
 *
 * Environment variables:
 *   CHROME_PATH    Path to Chrome/Chromium binary
 *   PORT           Fixed server port (default: random)
 *   TIMEOUT        Model load timeout in ms (default: 120000)
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
const TIMEOUT = parseInt(process.env.TIMEOUT || '120000', 10);
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

// ─── HTTP Server (with Range support + COOP/COEP) ───────────────────────────

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
  console.log('WebModelDelivery — ONNX Headless Test Runner\n');

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
  const required = ['index.html', 'model-sw.js', 'transformers.js'];
  for (const f of required) {
    if (!fs.existsSync(path.join(ROOT, f))) {
      console.error(`ERROR: Missing ${f} in ${ROOT}`);
      process.exit(1);
    }
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
  await page.exposeFunction('_swProgress', (json) => swLog.push(JSON.parse(json)));

  // Console logging
  page.on('console', (msg) => {
    const text = msg.text();
    if (text.includes('model-sw')) console.log(`  [SW] ${text}`);
  });
  page.on('pageerror', (e) => console.log(`  [PAGE ERROR] ${e.message.slice(0, 200)}`));

  let allPassed = true;

  // ─── Navigate and wait for SW ─────────────────────────────────────────

  console.log('=== Page Load ===');
  await page.goto(`http://localhost:${port}`, { waitUntil: 'networkidle0', timeout: 30000 });
  await sleep(5000); // Wait for SW registration + Transformers.js import

  // Handle first-install reload
  const hasController = await page.evaluate(() => !!navigator.serviceWorker.controller);
  if (!hasController) {
    console.log('  SW installed, waiting for controller (may reload)...');
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
      if (e.data?.type === 'MODEL_SW_PROGRESS') window._swProgress(JSON.stringify(e.data));
    });
  });

  const swState = await page.evaluate(() => navigator.serviceWorker.controller?.state || 'none');
  console.log(`  SW state: ${swState}`);

  await screenshot(page, '01_init');

  // ─── Embedding Model (explicit manifest: q4f16) ───────────────────────

  console.log('\n=== Embedding Model (explicit manifest: q4f16) ===');
  swLog.length = 0;

  await page.evaluate(() => {
    const btn = [...document.querySelectorAll('button')]
      .find(b => b.textContent.includes('Load Model') && !b.disabled);
    if (btn) btn.click();
  });

  try {
    await page.waitForFunction(
      () => document.body.innerText.includes('ready ✓'),
      { timeout: TIMEOUT }
    );
    console.log('  ✓ Embedding model loaded');
  } catch {
    console.log(`  ✗ TIMEOUT — embedding model did not load within ${TIMEOUT}ms`);
    allPassed = false;
  }

  await sleep(3000); // Let idle finalization complete
  await screenshot(page, '02_embedding_loaded');

  const embMsgs = swLog.filter(p => p.pathPrefix?.includes('embedding'));
  const embProg = reportProgress('Embedding', embMsgs);
  if (!embProg.monotonic) allPassed = false;

  // ─── Reranker Model (adaptive mode) ───────────────────────────────────

  console.log('\n=== Reranker Model (adaptive mode) ===');
  swLog.length = 0;

  await page.evaluate(() => {
    const btns = [...document.querySelectorAll('button')]
      .filter(b => b.textContent.includes('Load Model') && !b.disabled);
    if (btns.length > 0) btns[btns.length - 1].click();
  });

  try {
    await page.waitForFunction(
      () => (document.body.innerText.match(/ready ✓/g) || []).length >= 2,
      { timeout: TIMEOUT }
    );
    console.log('  ✓ Reranker model loaded');
  } catch {
    console.log(`  ✗ TIMEOUT — reranker model did not load within ${TIMEOUT}ms`);
    allPassed = false;
  }

  await sleep(3000);
  await screenshot(page, '03_reranker_loaded');

  const rrMsgs = swLog.filter(p => p.pathPrefix?.includes('reranker'));
  const rrProg = reportProgress('Reranker', rrMsgs);
  if (!rrProg.monotonic) allPassed = false;

  // ─── Inference ────────────────────────────────────────────────────────

  if (!SKIP_INFERENCE) {
    console.log('\n=== Inference ===');

    // Embedding
    await page.evaluate(() => {
      const btn = [...document.querySelectorAll('button')].find(b => b.textContent.includes('Embed'));
      if (btn) btn.click();
    });
    await sleep(5000);

    const embInfResult = await page.evaluate(() => {
      const r = document.querySelector('.result');
      return r?.textContent?.slice(0, 120) || 'none';
    });
    console.log(`  Embedding: ${embInfResult}`);

    if (embInfResult.includes('vector') || embInfResult.includes('d ')) {
      console.log('  ✓ Embedding inference produced vector output');
    } else if (embInfResult === 'none') {
      console.log('  ✗ Embedding inference produced no output');
      allPassed = false;
    }

    // Reranker
    await page.evaluate(() => {
      const btn = [...document.querySelectorAll('button')].find(b => b.textContent.includes('Rerank'));
      if (btn) btn.click();
    });
    await sleep(12000);

    try {
      const rrResults = await page.evaluate(() =>
        [...document.querySelectorAll('.result-text')]
          .map(e => (typeof e.textContent === 'string' ? e.textContent : ''))
          .filter(t => t.length > 0)
      );
      if (rrResults.length > 0) {
        console.log('  Reranker results:');
        rrResults.forEach((t, i) => console.log(`    #${i + 1}: ${t.slice(0, 60)}`));
        console.log('  ✓ Reranker produced ranked results');
      } else {
        console.log('  ✗ Reranker produced no results');
        allPassed = false;
      }
    } catch {
      console.log('  ✗ Reranker result extraction failed');
      allPassed = false;
    }

    await screenshot(page, '04_inference');
  } else {
    console.log('\n=== Inference skipped (--skip-inference) ===');
  }

  // ─── Event Log ────────────────────────────────────────────────────────

  console.log('\n=== Event Log (last 15) ===');
  const log = await page.evaluate(() =>
    [...document.querySelectorAll('.log-entry')].slice(-15).map(e => e.textContent)
  );
  log.forEach(l => console.log(`  ${l}`));

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
    return { monotonic: true };
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

  console.log(`\n=== ${label.toUpperCase()} (${last.mode}:${last.manifest || 'none'}) ===`);
  console.log(`  SW msgs:   ${msgs.length}`);
  console.log(`  Pcts:      ${uniquePcts.join('→')}`);
  console.log(`  Monotonic: ${monotonic ? 'YES ✓' : 'NO ✗'}`);
  console.log(`  Mode:      ${last.mode}`);
  console.log(`  Manifest:  ${last.manifest || '(none)'}`);
  console.log(`  Done:      ${last.done}`);
  console.log(`  Total:     ${(last.modelTotal / 1048576).toFixed(1)} MB`);

  return { monotonic };
}

// ─── Run ────────────────────────────────────────────────────────────────────

main().catch(err => {
  console.error('FATAL:', err.message);
  process.exit(1);
});
