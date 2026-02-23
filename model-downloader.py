#!/usr/bin/env python3
"""
model-downloader.py — Download model files from a filemap to a local folder.

Given a filemap.json URL (or local path), downloads files to a target directory.
Supports filtering by manifest name(s), SHA256 verification, and resume.

Usage:
  # Download all files from a CDN filemap:
  python model-downloader.py \
    https://cdn.jsdelivr.net/gh/user/models@v1/filemap.json \
    -o ./my-model

  # Download only files in the "q4f16" manifest:
  python model-downloader.py \
    https://cdn.jsdelivr.net/gh/user/models@v1/filemap.json \
    -m q4f16 -o ./my-model

  # Download files from multiple manifests:
  python model-downloader.py \
    https://cdn.jsdelivr.net/gh/user/models@v1/filemap.json \
    -m q4f16 -m quantized -o ./my-model

  # Download from a local flat-repo (reassemble shards → original files):
  python model-downloader.py /path/to/pkg-model -o ./my-model

  # List available manifests without downloading:
  python model-downloader.py https://example.com/filemap.json --list

Options:
  -o, --output       Target directory (default: current dir)
  -m, --manifest     Manifest name(s) to download (repeatable; omit for all)
  --list             List available manifests and exit
  --verify           Verify SHA256 checksums after download (default: on)
  --no-verify        Skip SHA256 verification
  --concurrency N    Parallel downloads (default: 4)
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def is_url(s):
    return s.startswith('http://') or s.startswith('https://')


def fetch_json(url):
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def read_json(path):
    with open(path) as f:
        return json.load(f)


def load_filemap(source):
    """Load filemap.json from a URL or local path."""
    if is_url(source):
        # If source is a direct URL to filemap.json
        if source.endswith('/filemap.json'):
            base_url = source.rsplit('/', 1)[0]
            return fetch_json(source), base_url
        # If source is a base URL, append filemap.json
        base_url = source.rstrip('/')
        return fetch_json(base_url + '/filemap.json'), base_url
    else:
        # Local path
        local = Path(source)
        if local.is_file() and local.name == 'filemap.json':
            return read_json(local), str(local.parent)
        elif local.is_dir():
            fmap = local / 'filemap.json'
            if not fmap.exists():
                sys.exit(f"No filemap.json found in {local}")
            return read_json(fmap), str(local)
        else:
            sys.exit(f"Cannot find filemap at: {source}")


def get_file_list(filemap, manifests):
    """Get list of virtual paths to download."""
    all_files = set(filemap.get('files', {}).keys())

    if not manifests:
        return sorted(all_files)

    result = set()
    available = filemap.get('manifests', {})
    for m in manifests:
        if m not in available:
            print(f"  ⚠ Manifest '{m}' not found. Available: {list(available.keys())}")
            continue
        manifest_files = available[m].get('files', [])
        result.update(manifest_files)
    return sorted(result)


def download_file(url, dest, expected_size=None):
    """Download a single file from URL."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Resume support: skip if already exists with correct size
    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        return 'cached'
    with urllib.request.urlopen(url) as resp:
        with open(dest, 'wb') as f:
            shutil.copyfileobj(resp, f)
    return 'downloaded'


def _cdn_name(entry):
    """Get the CDN filename from a filemap entry (handles field name variations)."""
    return entry.get('cdn_file') or entry.get('cdn') or entry.get('cdnFilename')


def _shard_name(shard):
    """Get the filename from a shard entry."""
    return shard.get('file') or shard.get('cdn_file') or shard.get('cdn')


def reassemble_local(base_path, entry, dest):
    """Reassemble a file from local shards."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == entry['size']:
        return 'cached'

    shards = entry.get('shards')
    if not shards:
        # Single file, direct copy
        cdn = _cdn_name(entry)
        if cdn:
            shutil.copy2(Path(base_path) / cdn, dest)
        else:
            return 'skip'
    else:
        # Reassemble shards
        with open(dest, 'wb') as f:
            for shard in shards:
                shard_path = Path(base_path) / _shard_name(shard)
                with open(shard_path, 'rb') as sf:
                    shutil.copyfileobj(sf, f)
    return 'assembled'


def download_from_cdn(base_url, entry, dest):
    """Download a file from CDN, reassembling shards if needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == entry['size']:
        return 'cached'

    shards = entry.get('shards')
    if not shards:
        cdn = _cdn_name(entry)
        download_file(f"{base_url}/{cdn}", dest, entry['size'])
    else:
        with open(dest, 'wb') as f:
            for shard in shards:
                url = f"{base_url}/{_shard_name(shard)}"
                with urllib.request.urlopen(url) as resp:
                    shutil.copyfileobj(resp, f)
    return 'downloaded'


def verify_sha256(filepath, expected):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest() == expected


def main():
    p = argparse.ArgumentParser(
        description='Download model files from a filemap.json source')
    p.add_argument('source',
        help='URL or local path to filemap.json (or directory containing it)')
    p.add_argument('-o', '--output', default='.',
        help='Target directory (default: current directory)')
    p.add_argument('-m', '--manifest', action='append', default=[],
        help='Manifest name(s) to download (repeatable; omit for all files)')
    p.add_argument('--list', action='store_true',
        help='List available manifests and exit')
    p.add_argument('--verify', dest='verify', action='store_true', default=True,
        help='Verify SHA256 checksums (default)')
    p.add_argument('--no-verify', dest='verify', action='store_false',
        help='Skip SHA256 verification')
    p.add_argument('--concurrency', type=int, default=4,
        help='Parallel downloads for CDN sources (default: 4)')
    args = p.parse_args()

    # Load filemap
    print(f"Loading filemap from: {args.source}")
    filemap, base = load_filemap(args.source)
    is_remote = is_url(base)

    version = filemap.get('version', '?')
    total_files = len(filemap.get('files', {}))
    manifests = filemap.get('manifests', {})
    print(f"  Filemap v{version}: {total_files} files, {len(manifests)} manifest(s)")

    # List mode
    if args.list:
        if not manifests:
            print("  (no manifests defined)")
        for name, mf in manifests.items():
            files = mf.get('files', [])
            total = sum(filemap['files'][f]['size'] for f in files if f in filemap['files'])
            print(f"  • {name}: {len(files)} files, {total / 1048576:.1f} MB")
        return

    # Determine files to download
    file_list = get_file_list(filemap, args.manifest)
    if not file_list:
        print("No files to download.")
        return

    total_size = sum(filemap['files'].get(f, {}).get('size', 0) for f in file_list)
    label = ', '.join(args.manifest) if args.manifest else 'all'
    print(f"  Downloading [{label}]: {len(file_list)} files, {total_size / 1048576:.1f} MB")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download/assemble files
    stats = {'cached': 0, 'downloaded': 0, 'assembled': 0, 'failed': 0, 'verified': 0}

    def process_file(vp):
        entry = filemap['files'].get(vp)
        if not entry:
            return vp, 'skip'
        dest = out_dir / vp
        try:
            if is_remote:
                status = download_from_cdn(base, entry, dest)
            else:
                status = reassemble_local(base, entry, dest)
            return vp, status
        except Exception as e:
            return vp, f'error: {e}'

    workers = args.concurrency if is_remote else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_file, vp): vp for vp in file_list}
        for i, future in enumerate(as_completed(futures), 1):
            vp, status = future.result()
            if status.startswith('error'):
                stats['failed'] += 1
                print(f"  ✗ {vp}: {status}")
            else:
                stats[status] = stats.get(status, 0) + 1
                size_mb = filemap['files'][vp]['size'] / 1048576
                print(f"  [{i}/{len(file_list)}] {vp} ({size_mb:.1f} MB) — {status}")

    # Copy filemap.json
    filemap_dest = out_dir / 'filemap.json'
    if is_remote:
        with open(filemap_dest, 'w') as f:
            json.dump(filemap, f, indent=2)
    else:
        src_filemap = Path(base) / 'filemap.json'
        shutil.copy2(src_filemap, filemap_dest)
    print(f"  filemap.json → {filemap_dest}")

    # Verify
    if args.verify:
        print("\nVerifying SHA256 checksums...")
        for vp in file_list:
            entry = filemap['files'].get(vp, {})
            sha = entry.get('sha256')
            dest = out_dir / vp
            if sha and dest.exists():
                if verify_sha256(dest, sha):
                    stats['verified'] += 1
                else:
                    print(f"  ✗ SHA256 MISMATCH: {vp}")
                    stats['failed'] += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"✓ Complete: {stats.get('downloaded',0) + stats.get('assembled',0)} new, "
          f"{stats['cached']} cached, {stats.get('verified',0)} verified, "
          f"{stats['failed']} failed")
    print(f"  Output: {out_dir.resolve()}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
