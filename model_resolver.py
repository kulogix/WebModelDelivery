#!/usr/bin/env python3
"""
model_resolver.py — Python equivalent of model-sw.js

Transparently resolves model files from filemap.json-based CDN sharded repos
OR local flat-repo directories for Python ML libraries.

Supports BOTH remote CDN sources and local flat-repo directories.

─── Usage: Direct Resolve ──────────────────────────────────────────────────

    from model_resolver import ModelResolver

    resolver = ModelResolver(cache_dir='./.model-cache')

    # From CDN:
    local_dir = resolver.resolve(
        'https://cdn.jsdelivr.net/gh/user/cdn-embedding@v1',
        manifest='q4f16',
    )

    # From local flat repo:
    local_dir = resolver.resolve(
        '/path/to/pkg-embedding',
        manifest='q4f16',
    )

    # Now use with any library:
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModel.from_pretrained(local_dir)

─── Usage: GGUF Resolve ────────────────────────────────────────────────────

    from model_resolver import ModelResolver, resolve_gguf

    # One-shot from local flat repo:
    gguf_paths = resolve_gguf('/path/to/pkg-gemma3', manifest='q4_0')
    from llama_cpp import Llama
    llm = Llama(model_path=gguf_paths[0])

─── Usage: Monkey-Patch huggingface_hub ────────────────────────────────────

    resolver = ModelResolver()
    resolver.patch_hf(
        model_id='my-org/my-model',
        cdn_base='/path/to/pkg-model',     # local or CDN URL
        manifest='q4f16',
    )
    from transformers import AutoModel
    model = AutoModel.from_pretrained('my-org/my-model')

─── Usage: Local File Server ───────────────────────────────────────────────

    resolver = ModelResolver()
    server = resolver.serve('/path/to/pkg-model', manifest='q4f16', port=8787)
    # Access at http://localhost:8787/config.json, etc.
    server.shutdown()
"""

import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


def _is_local_path(s: str) -> bool:
    """Determine if a string is a local filesystem path vs a URL."""
    if not s:
        return False
    if s.startswith(('/', './', '../')):
        return True
    if len(s) >= 2 and s[1] == ':' and s[0].isalpha():
        return True
    if s.startswith('file://'):
        return True
    if s.startswith(('http://', 'https://')):
        return False
    return True


def _to_local_path(s: str) -> str:
    if s.startswith('file://'):
        from urllib.parse import urlparse
        return urlparse(s).path
    return s


# ─── ModelResolver ──────────────────────────────────────────────────────────

class ModelResolver:
    """
    Resolves model files from filemap.json CDN sharded repos or local flat repos.
    """

    def __init__(
        self,
        cache_dir: str = './.model-cache',
        verify_sha256: bool = False,
        concurrency: int = 4,
        retries: int = 3,
        chunk_read_size: int = 8 * 1024 * 1024,
    ):
        self.cache_dir = Path(cache_dir).resolve()
        self.verify_sha256 = verify_sha256
        self.concurrency = concurrency
        self.retries = retries
        self.chunk_read_size = chunk_read_size

        self._filemaps: Dict[str, dict] = {}
        self._filemap_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    # ─── Primary API: resolve to local directory ──────────────────────────

    def resolve(
        self,
        source: str,
        manifest: Optional[str] = None,
        on_progress: Optional[Callable[[dict], None]] = None,
    ) -> str:
        """
        Download/reassemble all files from a source into a local directory.

        Args:
            source: CDN URL root OR local flat-repo directory path
            manifest: Manifest name (e.g., 'q4f16'). None = all files.
            on_progress: Callback with {percent, loaded, total, file, done}

        Returns:
            Absolute path to local directory with reassembled model files.
        """
        is_local = _is_local_path(source)
        source_key = str(Path(_to_local_path(source)).resolve()) if is_local else source.rstrip('/')
        filemap = self._load_filemap(source_key, is_local)
        if not filemap:
            raise RuntimeError(f"Failed to load filemap from {source}")

        file_list = self._get_file_list(filemap, manifest)
        out_dir = self._cache_path_for_source(source_key, manifest)
        out_dir.mkdir(parents=True, exist_ok=True)

        total_bytes = sum(filemap['files'].get(vp, {}).get('size', 0) for vp in file_list)
        loaded_bytes = 0

        for vp in file_list:
            entry = filemap['files'].get(vp)
            if not entry:
                continue

            out_path = out_dir / vp
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Skip if already exists with correct size
            if out_path.exists() and out_path.stat().st_size == entry['size']:
                loaded_bytes += entry['size']
                if on_progress:
                    on_progress({
                        'percent': min(100, round(loaded_bytes / total_bytes * 100)) if total_bytes else 0,
                        'loaded': loaded_bytes, 'total': total_bytes,
                        'file': vp, 'done': False,
                    })
                continue

            def _on_bytes(b):
                nonlocal loaded_bytes
                loaded_bytes += b
                if on_progress:
                    on_progress({
                        'percent': min(100, round(loaded_bytes / total_bytes * 100)) if total_bytes else 0,
                        'loaded': loaded_bytes, 'total': total_bytes,
                        'file': vp, 'done': False,
                    })

            self._reassemble_file(source_key, is_local, vp, entry, str(out_path), _on_bytes)

        if on_progress:
            on_progress({
                'percent': 100, 'loaded': total_bytes, 'total': total_bytes,
                'file': '', 'done': True,
            })

        return str(out_dir)

    def resolve_files(
        self,
        source: str,
        manifest: Optional[str] = None,
        on_progress: Optional[Callable[[dict], None]] = None,
    ) -> Dict[str, str]:
        """Like resolve(), returns {virtual_path: absolute_local_path}."""
        local_dir = self.resolve(source, manifest=manifest, on_progress=on_progress)
        is_local = _is_local_path(source)
        source_key = str(Path(_to_local_path(source)).resolve()) if is_local else source.rstrip('/')
        filemap = self._filemaps.get(source_key)
        file_list = self._get_file_list(filemap, manifest)
        return {vp: os.path.join(local_dir, vp) for vp in file_list}

    # ─── Monkey-patch huggingface_hub ─────────────────────────────────────

    def patch_hf(
        self,
        model_id: str,
        cdn_base: str,
        manifest: Optional[str] = None,
        on_progress: Optional[Callable[[dict], None]] = None,
    ):
        """
        Monkey-patch huggingface_hub so that from_pretrained(model_id) loads
        from CDN shards or local flat repo instead of HuggingFace Hub.

        Works with both CDN URLs and local paths.
        """
        local_dir = self.resolve(cdn_base, manifest=manifest, on_progress=on_progress)

        try:
            import huggingface_hub
        except ImportError:
            raise ImportError("huggingface_hub not installed. pip install huggingface_hub")

        original_snapshot_download = huggingface_hub.snapshot_download
        original_hf_hub_download = huggingface_hub.hf_hub_download

        def patched_snapshot_download(repo_id, *args, **kwargs):
            if repo_id == model_id:
                return local_dir
            return original_snapshot_download(repo_id, *args, **kwargs)

        def patched_hf_hub_download(repo_id, filename, *args, **kwargs):
            if repo_id == model_id:
                local_path = os.path.join(local_dir, filename)
                if os.path.exists(local_path):
                    return local_path
            return original_hf_hub_download(repo_id, filename, *args, **kwargs)

        huggingface_hub.snapshot_download = patched_snapshot_download
        huggingface_hub.hf_hub_download = patched_hf_hub_download

        if hasattr(huggingface_hub, 'cached_download'):
            original_cached = huggingface_hub.cached_download
            def patched_cached(url_or_filename, *args, **kwargs):
                if model_id in str(url_or_filename):
                    filename = str(url_or_filename).split('/')[-1]
                    local_path = os.path.join(local_dir, filename)
                    if os.path.exists(local_path):
                        return local_path
                return original_cached(url_or_filename, *args, **kwargs)
            huggingface_hub.cached_download = patched_cached

    # ─── Local HTTP File Server ───────────────────────────────────────────

    def serve(
        self,
        source: str,
        manifest: Optional[str] = None,
        port: int = 0,
        host: str = '127.0.0.1',
        on_progress: Optional[Callable[[dict], None]] = None,
    ) -> 'ModelFileServer':
        """
        Start a local HTTP server serving the reassembled model files.
        Works with both CDN URLs and local flat-repo paths.
        """
        local_dir = self.resolve(source, manifest=manifest, on_progress=on_progress)
        return ModelFileServer(local_dir, host=host, port=port)

    # ─── Filemap Inspection ──────────────────────────────────────────────

    def get_filemap(self, source: str) -> Optional[dict]:
        is_local = _is_local_path(source)
        key = str(Path(_to_local_path(source)).resolve()) if is_local else source.rstrip('/')
        return self._load_filemap(key, is_local)

    def list_manifests(self, source: str) -> Dict[str, dict]:
        filemap = self.get_filemap(source)
        if not filemap:
            return {}
        manifests = filemap.get('manifests', {})
        return {
            name: {
                'files': len(m.get('files', [])),
                'size': m.get('size', 0),
                'size_mb': round(m.get('size', 0) / 1048576, 1),
            }
            for name, m in manifests.items()
        }

    def get_gguf_metadata(self, source: str) -> dict:
        filemap = self.get_filemap(source)
        return filemap.get('gguf_metadata', {}) if filemap else {}

    # ─── Cache Management ────────────────────────────────────────────────

    def clear_cache(self):
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self._filemaps.clear()

    def get_cache_stats(self) -> dict:
        total_bytes = 0
        total_files = 0
        if self.cache_dir.exists():
            for f in self.cache_dir.rglob('*'):
                if f.is_file():
                    total_files += 1
                    total_bytes += f.stat().st_size
        return {
            'cache_dir': str(self.cache_dir),
            'files': total_files, 'bytes': total_bytes,
            'mb': round(total_bytes / 1048576, 1),
        }

    # ─── Internal: Filemap Loading ────────────────────────────────────────

    def _load_filemap(self, source_key: str, is_local: bool = False) -> Optional[dict]:
        if source_key in self._filemaps:
            return self._filemaps[source_key]

        with self._global_lock:
            if source_key not in self._filemap_locks:
                self._filemap_locks[source_key] = threading.Lock()

        with self._filemap_locks[source_key]:
            if source_key in self._filemaps:
                return self._filemaps[source_key]

            try:
                if is_local:
                    fp = os.path.join(source_key, 'filemap.json')
                    if not os.path.exists(fp):
                        raise FileNotFoundError(f"filemap.json not found in {source_key}")
                    data = json.loads(Path(fp).read_text())
                else:
                    cache_path = self.cache_dir / 'filemaps' / (
                        hashlib.sha256(source_key.encode()).hexdigest()[:16] + '.json'
                    )
                    if cache_path.exists():
                        data = json.loads(cache_path.read_text())
                    else:
                        data = json.loads(self._download_text(f"{source_key}/filemap.json"))
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        cache_path.write_text(json.dumps(data, indent=2))

                self._filemaps[source_key] = data
                return data

            except Exception as e:
                print(f"[model-resolver] Failed to load filemap from {source_key}: {e}",
                      file=sys.stderr)
                return None

    # ─── Internal: File List ─────────────────────────────────────────────

    def _get_file_list(self, filemap: dict, manifest: Optional[str]) -> List[str]:
        if manifest and 'manifests' in filemap and manifest in filemap['manifests']:
            return filemap['manifests'][manifest]['files']
        return list(filemap.get('files', {}).keys())

    # ─── Internal: File Reassembly ────────────────────────────────────────

    def _reassemble_file(self, source_key: str, is_local: bool, vp: str,
                          entry: dict, out_path: str, on_bytes=None):
        if not entry.get('shards'):
            cdn_file = entry.get('cdn_file', vp)
            if is_local:
                buf = self._read_local(source_key, cdn_file)
            else:
                buf = self._download_shard(f"{source_key}/{cdn_file}")
            Path(out_path).write_bytes(buf)
            if on_bytes:
                on_bytes(entry['size'])
        else:
            with open(out_path, 'wb') as f:
                for shard in entry['shards']:
                    if is_local:
                        buf = self._read_local(source_key, shard['file'])
                    else:
                        buf = self._download_shard(f"{source_key}/{shard['file']}")
                    f.seek(shard['offset'])
                    f.write(buf)
                    if on_bytes:
                        on_bytes(shard['size'])

        if self.verify_sha256 and 'sha256' in entry:
            actual = self._sha256_file(out_path)
            if actual != entry['sha256']:
                os.unlink(out_path)
                raise RuntimeError(
                    f"SHA256 mismatch for {vp}: expected {entry['sha256']}, got {actual}"
                )

    # ─── Internal: Local File Reading ────────────────────────────────────

    def _read_local(self, base: str, filename: str) -> bytes:
        fp = os.path.join(base, filename)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"[model-resolver] Local file not found: {fp}")
        return Path(fp).read_bytes()

    # ─── Internal: Shard Download & Cache (CDN mode) ─────────────────────

    def _download_shard(self, url: str) -> bytes:
        cache_path = self._shard_cache_path(url)
        if cache_path.exists():
            return cache_path.read_bytes()

        last_err = None
        for attempt in range(self.retries):
            try:
                data = self._download_bytes(url)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(data)
                return data
            except Exception as e:
                last_err = e
                if attempt < self.retries - 1:
                    time.sleep(1 * (attempt + 1))

        raise RuntimeError(f"Failed to download {url} after {self.retries} attempts: {last_err}")

    def _shard_cache_path(self, url: str) -> Path:
        h = hashlib.sha256(url.encode()).hexdigest()[:16]
        basename = url.split('/')[-1]
        return self.cache_dir / 'shards' / f"{h}_{basename}"

    def _cache_path_for_source(self, source_key: str, manifest: Optional[str]) -> Path:
        h = hashlib.sha256(source_key.encode()).hexdigest()[:12]
        suffix = f"_{manifest}" if manifest else ""
        return self.cache_dir / 'resolved' / f"{h}{suffix}"

    # ─── Internal: HTTP Downloads ─────────────────────────────────────────

    def _download_bytes(self, url: str) -> bytes:
        req = Request(url, headers={'User-Agent': 'ModelResolver/1.0'})
        with urlopen(req, timeout=60) as resp:
            return resp.read()

    def _download_text(self, url: str) -> str:
        return self._download_bytes(url).decode('utf-8')

    def _sha256_file(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(self.chunk_read_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()


# ─── Local File Server ──────────────────────────────────────────────────────

class ModelFileServer:
    """Local HTTP server that serves reassembled model files with Range support."""

    def __init__(self, root_dir: str, host: str = '127.0.0.1', port: int = 0):
        self.root_dir = os.path.abspath(root_dir)
        _root = self.root_dir

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=_root, **kwargs)

            def do_GET(self):
                range_header = self.headers.get('Range')
                if range_header:
                    self._handle_range(range_header)
                    return
                super().do_GET()

            def _handle_range(self, range_header):
                path = self.translate_path(self.path)
                if not os.path.isfile(path):
                    self.send_error(404)
                    return
                m = re.match(r'bytes=(\d+)-(\d*)', range_header)
                if not m:
                    super().do_GET()
                    return
                file_size = os.path.getsize(path)
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else file_size - 1
                if start > end or start >= file_size:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{file_size}')
                    self.end_headers()
                    return
                length = end - start + 1
                self.send_response(206)
                self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                self.send_header('Content-Length', length)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Accept-Ranges', 'bytes')
                self.end_headers()
                with open(path, 'rb') as f:
                    f.seek(start)
                    self.wfile.write(f.read(length))

            def end_headers(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Accept-Ranges', 'bytes')
                super().end_headers()

            def log_message(self, format, *args):
                pass

        self._server = HTTPServer((host, port), Handler)
        self.port = self._server.server_address[1]
        self.host = host
        self.url = f"http://{host}:{self.port}"

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def shutdown(self):
        self._server.shutdown()
        self._thread.join(timeout=5)


# ─── Convenience Functions ──────────────────────────────────────────────────

def resolve_model(
    source: str,
    manifest: Optional[str] = None,
    cache_dir: str = './.model-cache',
    on_progress: Optional[Callable[[dict], None]] = None,
    verify_sha256: bool = False,
) -> str:
    """One-shot: resolve a model from CDN or local flat repo to local directory."""
    resolver = ModelResolver(cache_dir=cache_dir, verify_sha256=verify_sha256)
    return resolver.resolve(source, manifest=manifest, on_progress=on_progress)


def resolve_gguf(
    source: str,
    manifest: Optional[str] = None,
    cache_dir: str = './.model-cache',
    on_progress: Optional[Callable[[dict], None]] = None,
) -> List[str]:
    """One-shot: resolve GGUF model and return list of .gguf file paths."""
    resolver = ModelResolver(cache_dir=cache_dir)
    files = resolver.resolve_files(source, manifest=manifest, on_progress=on_progress)
    gguf_files = sorted([p for vp, p in files.items() if vp.endswith('.gguf')])
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf files found in manifest '{manifest}' at {source}")
    return gguf_files


# ─── CLI ────────────────────────────────────────────────────────────────────

def _cli_progress(info: dict):
    pct = info['percent']
    loaded_mb = info['loaded'] / 1048576
    total_mb = info['total'] / 1048576
    file = info.get('file', '')
    if info['done']:
        print(f"\r✓ Complete — {loaded_mb:.1f} MB                              ")
    else:
        bar = '█' * (pct // 5) + '░' * (20 - pct // 5)
        print(f"\r  {bar} {pct:3d}% — {loaded_mb:.1f}/{total_mb:.1f} MB — {file[:40]}", end='', flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Download and reassemble model files from CDN or local flat repos',
    )
    sub = parser.add_subparsers(dest='command')

    p_resolve = sub.add_parser('resolve', help='Resolve model to local dir')
    p_resolve.add_argument('source', help='CDN URL root or local flat-repo path')
    p_resolve.add_argument('--manifest', '-m', help='Manifest name')
    p_resolve.add_argument('--cache-dir', default='./.model-cache')
    p_resolve.add_argument('--verify-sha256', action='store_true')
    p_resolve.add_argument('--quiet', '-q', action='store_true')

    p_list = sub.add_parser('list', help='List manifests')
    p_list.add_argument('source', help='CDN URL root or local flat-repo path')
    p_list.add_argument('--cache-dir', default='./.model-cache')

    p_serve = sub.add_parser('serve', help='Start local file server')
    p_serve.add_argument('source', help='CDN URL root or local flat-repo path')
    p_serve.add_argument('--manifest', '-m')
    p_serve.add_argument('--port', '-p', type=int, default=8787)
    p_serve.add_argument('--cache-dir', default='./.model-cache')

    p_cache = sub.add_parser('cache-stats', help='Show cache statistics')
    p_cache.add_argument('--cache-dir', default='./.model-cache')

    p_clear = sub.add_parser('clear-cache', help='Clear all cached files')
    p_clear.add_argument('--cache-dir', default='./.model-cache')

    args = parser.parse_args()

    if args.command == 'resolve':
        resolver = ModelResolver(cache_dir=args.cache_dir, verify_sha256=args.verify_sha256)
        progress_fn = None if args.quiet else _cli_progress
        local_dir = resolver.resolve(args.source, manifest=args.manifest, on_progress=progress_fn)
        print(f"\nResolved to: {local_dir}")

    elif args.command == 'list':
        resolver = ModelResolver(cache_dir=args.cache_dir)
        manifests = resolver.list_manifests(args.source)
        if not manifests:
            print("No manifests found (or filemap load failed)")
        else:
            print(f"Manifests in {args.source}:\n")
            for name, info in sorted(manifests.items()):
                print(f"  {name:20s}  {info['files']:3d} files  {info['size_mb']:8.1f} MB")
        gguf = resolver.get_gguf_metadata(args.source)
        if gguf:
            print(f"\nGGUF metadata:")
            for key, meta in gguf.items():
                cls = meta.get('classification', '?')
                arch = meta.get('architecture', '?')
                quant = meta.get('quantization', '?')
                ctx = meta.get('context_length', '?')
                print(f"  [{cls}] {key}: arch={arch} quant={quant} ctx={ctx}")

    elif args.command == 'serve':
        resolver = ModelResolver(cache_dir=args.cache_dir)
        print(f"Resolving model from {args.source}...")
        server = resolver.serve(args.source, manifest=args.manifest, port=args.port,
                                 on_progress=_cli_progress)
        print(f"\nServing at: {server.url}")
        print("Press Ctrl+C to stop.\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            server.shutdown()

    elif args.command == 'cache-stats':
        resolver = ModelResolver(cache_dir=args.cache_dir)
        stats = resolver.get_cache_stats()
        print(f"Cache: {stats['cache_dir']}")
        print(f"Files: {stats['files']}")
        print(f"Size:  {stats['mb']:.1f} MB ({stats['bytes']:,} bytes)")

    elif args.command == 'clear-cache':
        resolver = ModelResolver(cache_dir=args.cache_dir)
        resolver.clear_cache()
        print(f"Cache cleared: {args.cache_dir}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
