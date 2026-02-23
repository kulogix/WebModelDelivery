#!/usr/bin/env node
/**
 * serve.mjs — Development server for WebModelDelivery test harness
 *
 * Features:
 *   - HTTP Range requests (required for ONNX Runtime and wllama)
 *   - Correct MIME types for .mjs, .wasm, .json, .gguf
 *   - Service-Worker-Allowed header
 *   - CORS headers
 *   - COOP/COEP headers (required for SharedArrayBuffer / multi-thread wllama)
 *   - Random port by default (avoids conflicts), with retry on EADDRINUSE
 *
 * Usage:
 *   node serve.mjs                    # Serve current directory, random port
 *   node serve.mjs ./my-harness       # Serve specific directory
 *   PORT=8080 node serve.mjs          # Fixed port (retries nearby if busy)
 */

import http from 'http';
import fs from 'fs';
import path from 'path';
import net from 'net';

const ROOT = path.resolve(process.argv[2] || '.');
const REQUESTED_PORT = parseInt(process.env.PORT || '0', 10); // 0 = OS picks random
const MAX_RETRIES = 10;

const MIME = {
  '.html':  'text/html; charset=utf-8',
  '.js':    'text/javascript',
  '.mjs':   'text/javascript',         // CRITICAL for ONNX Runtime
  '.json':  'application/json',
  '.wasm':  'application/wasm',         // CRITICAL for WebAssembly
  '.css':   'text/css',
  '.onnx':  'application/octet-stream',
  '.model': 'application/octet-stream',
  '.gguf':  'application/octet-stream',
  '.txt':   'text/plain',
  '.md':    'text/markdown',
  '.png':   'image/png',
  '.svg':   'image/svg+xml',
};

// ─── Port availability check ────────────────────────────────────────────────

function isPortFree(port) {
  return new Promise((resolve) => {
    const tester = net.createServer()
      .once('error', () => resolve(false))
      .once('listening', () => tester.close(() => resolve(true)))
      .listen(port);
  });
}

async function findFreePort(startPort) {
  // Port 0 = let the OS choose (always succeeds)
  if (startPort === 0) return 0;

  // User requested a specific port — check it, then try nearby
  for (let offset = 0; offset < MAX_RETRIES; offset++) {
    const port = startPort + offset;
    if (port > 65535) break;
    if (await isPortFree(port)) return port;
    if (offset === 0) {
      console.log(`  ⚠ Port ${port} is in use, trying nearby...`);
    }
  }
  console.log(`  ⚠ Ports ${startPort}–${startPort + MAX_RETRIES - 1} all in use, falling back to random port`);
  return 0; // Fall back to OS-assigned random port
}

// ─── HTTP Server ────────────────────────────────────────────────────────────

function createServer() {
  return http.createServer((req, res) => {
    const url = new URL(req.url, `http://localhost`);
    let filePath = path.join(ROOT, decodeURIComponent(url.pathname));

    // Default to index.html
    if (url.pathname === '/' || url.pathname === '') {
      filePath = path.join(ROOT, 'index.html');
    }

    // Resolve symlinks, check existence
    try { filePath = fs.realpathSync(filePath); } catch {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end(`Not found: ${url.pathname}`);
      return;
    }

    if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
      // Try index.html in directory
      const idx = path.join(filePath, 'index.html');
      if (fs.existsSync(idx)) { filePath = idx; }
      else { res.writeHead(404); res.end('Not found'); return; }
    }

    const ext = path.extname(filePath).toLowerCase();
    const mime = MIME[ext] || 'application/octet-stream';
    const stat = fs.statSync(filePath);

    // ─── Range request handling (required for ONNX Runtime) ───
    const rangeHeader = req.headers.range;
    if (rangeHeader) {
      const match = rangeHeader.match(/bytes=(\d+)-(\d*)/);
      if (match) {
        const start = parseInt(match[1], 10);
        const end = match[2] ? parseInt(match[2], 10) : stat.size - 1;

        if (start > end || start >= stat.size) {
          res.writeHead(416, { 'Content-Range': `bytes */${stat.size}` });
          res.end();
          return;
        }

        res.writeHead(206, {
          'Content-Range': `bytes ${start}-${end}/${stat.size}`,
          'Accept-Ranges': 'bytes',
          'Content-Length': end - start + 1,
          'Content-Type': mime,
          'Access-Control-Allow-Origin': '*',
          'Service-Worker-Allowed': '/',
          'Cross-Origin-Embedder-Policy': 'require-corp',
          'Cross-Origin-Opener-Policy': 'same-origin',
        });
        fs.createReadStream(filePath, { start, end }).pipe(res);
        return;
      }
    }

    // ─── Normal request ───
    res.writeHead(200, {
      'Content-Type': mime,
      'Content-Length': stat.size,
      'Accept-Ranges': 'bytes',
      'Access-Control-Allow-Origin': '*',
      'Service-Worker-Allowed': '/',
      'Cache-Control': 'no-cache',      // Dev convenience: no stale files
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    });
    fs.createReadStream(filePath).pipe(res);
  });
}

// ─── Startup ────────────────────────────────────────────────────────────────

async function main() {
  // Verify root directory exists
  if (!fs.existsSync(ROOT)) {
    console.error(`ERROR: Root directory does not exist: ${ROOT}`);
    process.exit(1);
  }

  const indexFile = fs.existsSync(path.join(ROOT, 'index.html')) ? 'index.html' : null;
  if (!indexFile) {
    console.log(`  ⚠ No index.html in ${ROOT} — you'll need to navigate to a specific file.`);
  }

  const port = await findFreePort(REQUESTED_PORT);
  const server = createServer();

  server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
      console.error(`ERROR: Port ${port} is unexpectedly in use. Try a different port or use PORT=0 for random.`);
      process.exit(1);
    }
    throw err;
  });

  server.listen(port, () => {
    const actualPort = server.address().port;
    const url = `http://localhost:${actualPort}`;
    console.log(`WebModelDelivery dev server`);
    console.log(`  Root:  ${ROOT}`);
    console.log(`  URL:   ${url}`);
    if (indexFile) console.log(`  Open:  ${url}/${indexFile}`);
    console.log(`  Range: ✓  MIME: ✓  CORS: ✓  SW-Allowed: ✓  COOP/COEP: ✓`);
    console.log(`\nPress Ctrl+C to stop.`);
  });

  // Graceful shutdown
  const shutdown = () => {
    console.log('\nShutting down...');
    server.close(() => process.exit(0));
    // Force exit after 2s if connections hang
    setTimeout(() => process.exit(0), 2000);
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

main();
