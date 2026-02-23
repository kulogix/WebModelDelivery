#!/usr/bin/env python3
"""
cors_server.py — Development server for WebModelDelivery harnesses.

Adds headers required for Service Worker + multi-thread wllama:
  - CORS (Access-Control-Allow-Origin: *)
  - COOP (Cross-Origin-Opener-Policy: same-origin)
  - COEP (Cross-Origin-Embedder-Policy: credentialless)
  - Service-Worker-Allowed: /

Usage:
  python3 cors_server.py          # port 8000
  python3 cors_server.py 3000     # port 3000

Note: COOP/COEP enables SharedArrayBuffer for multi-thread wllama.
      Using 'credentialless' (not 'require-corp') so cross-origin CDN
      scripts (Tailwind, Alpine) load without needing CORP headers.
      The model-sw.js wraps cross-origin CDN responses in same-origin
      Response objects for full compatibility.
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import sys


class CORSCOEPHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type, Range')
        # COOP/COEP (required for SharedArrayBuffer → multi-thread wllama)
        # 'credentialless' allows cross-origin CDN scripts without CORP headers
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'credentialless')
        # Allow SW to control all paths
        self.send_header('Service-Worker-Allowed', '/')
        SimpleHTTPRequestHandler.end_headers(self)

    def do_OPTIONS(self):
        self.send_response(200, 'ok')
        self.end_headers()


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    httpd = HTTPServer(('0.0.0.0', port), CORSCOEPHandler)
    print(f'Serving at http://localhost:{port}')
    print(f'  CORS: ✓  COOP: same-origin  COEP: credentialless')
    print(f'  SharedArrayBuffer: ✓  CDN scripts: ✓  SW: ✓')
    httpd.serve_forever()
