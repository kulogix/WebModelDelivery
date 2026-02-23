#!/usr/bin/env python3
"""
Test suite for model_resolver.py — validates local flat-repo support.
Tests all three packaged models: ONNX embedding, ONNX reranker, GGUF LLM.
"""
import sys, os, json, hashlib, time
sys.path.insert(0, '/home/claude/WebModelDelivery')
from model_resolver import ModelResolver, resolve_model, resolve_gguf

PKG_EMBEDDING = '/home/claude/obtained/pkg-embedding'
PKG_RERANKER  = '/home/claude/obtained/pkg-reranker'
PKG_GEMMA3    = '/home/claude/obtained/pkg-gemma3'
CACHE_DIR     = '/tmp/test-resolver-cache-py'

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f'  ✓ {name}')
        passed += 1
    except Exception as e:
        print(f'  ✗ {name}: {e}')
        failed += 1

def assert_eq(a, b, msg=''):
    if a != b:
        raise AssertionError(f'{msg}: {a!r} != {b!r}')

def assert_true(v, msg=''):
    if not v:
        raise AssertionError(msg or 'assertion failed')

# ─── Clean start ─────────────────────────────────────────────────────────

import shutil
if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)

resolver = ModelResolver(cache_dir=CACHE_DIR, verify_sha256=True)

# ─── Test 1: Filemap loading from local path ────────────────────────────

print('\n=== Filemap Loading ===')

def test_filemap_load_embedding():
    fm = resolver.get_filemap(PKG_EMBEDDING)
    assert_true(fm is not None, 'filemap is None')
    assert_eq(fm['version'], 5, 'version')
    assert_true('files' in fm, 'no files key')
    assert_true(len(fm['files']) == 10, f'expected 10 files, got {len(fm["files"])}')

test('Load embedding filemap from local path', test_filemap_load_embedding)

def test_filemap_load_reranker():
    fm = resolver.get_filemap(PKG_RERANKER)
    assert_true(fm is not None)
    assert_true('onnx/model_quantized.onnx' in fm['files'])

test('Load reranker filemap from local path', test_filemap_load_reranker)

def test_filemap_load_gguf():
    fm = resolver.get_filemap(PKG_GEMMA3)
    assert_true(fm is not None)
    assert_true('gemma-3-1b-it-q4_0.gguf' in fm['files'])

test('Load GGUF filemap from local path', test_filemap_load_gguf)

# ─── Test 2: Manifest listing ───────────────────────────────────────────

print('\n=== Manifest Listing ===')

def test_list_manifests_embedding():
    manifests = resolver.list_manifests(PKG_EMBEDDING)
    assert_true('q4f16' in manifests, f'q4f16 not in {manifests.keys()}')
    assert_true(manifests['q4f16']['size_mb'] > 100, 'manifest too small')

test('List manifests for embedding', test_list_manifests_embedding)

def test_list_manifests_reranker():
    manifests = resolver.list_manifests(PKG_RERANKER)
    assert_true('quantized' in manifests, f'quantized not in {manifests.keys()}')

test('List manifests for reranker', test_list_manifests_reranker)

def test_list_manifests_gguf():
    manifests = resolver.list_manifests(PKG_GEMMA3)
    assert_true('q4_0' in manifests, f'q4_0 not in {manifests.keys()}')

test('List manifests for GGUF', test_list_manifests_gguf)

# ─── Test 3: Resolve with manifest (ONNX embedding) ─────────────────────

print('\n=== Resolve: ONNX Embedding (q4f16) ===')

progress_log = []

def test_resolve_embedding():
    progress_log.clear()
    local_dir = resolver.resolve(
        PKG_EMBEDDING,
        manifest='q4f16',
        on_progress=lambda p: progress_log.append(p),
    )
    assert_true(os.path.isdir(local_dir), f'not a dir: {local_dir}')
    # Check key files exist with correct sizes
    fm = resolver.get_filemap(PKG_EMBEDDING)
    for vp in fm['manifests']['q4f16']['files']:
        fp = os.path.join(local_dir, vp)
        assert_true(os.path.exists(fp), f'missing: {vp}')
        assert_eq(os.path.getsize(fp), fm['files'][vp]['size'], f'size mismatch: {vp}')

test('Resolve embedding to local dir', test_resolve_embedding)

def test_progress_embedding():
    assert_true(len(progress_log) > 0, 'no progress events')
    last = progress_log[-1]
    assert_eq(last['done'], True, 'last event not done')
    assert_eq(last['percent'], 100, 'last event not 100%')
    # Check monotonicity
    pcts = [p['percent'] for p in progress_log]
    for i in range(1, len(pcts)):
        assert_true(pcts[i] >= pcts[i-1], f'non-monotonic at {i}: {pcts[i-1]} > {pcts[i]}')

test('Progress tracking (embedding)', test_progress_embedding)

# ─── Test 4: Resolve with manifest (ONNX reranker) ──────────────────────

print('\n=== Resolve: ONNX Reranker (quantized) ===')

def test_resolve_reranker():
    local_dir = resolver.resolve(PKG_RERANKER, manifest='quantized')
    fm = resolver.get_filemap(PKG_RERANKER)
    onnx_path = os.path.join(local_dir, 'onnx/model_quantized.onnx')
    assert_true(os.path.exists(onnx_path), 'onnx model missing')
    assert_eq(os.path.getsize(onnx_path), fm['files']['onnx/model_quantized.onnx']['size'])

test('Resolve reranker to local dir', test_resolve_reranker)

# ─── Test 5: Resolve GGUF ───────────────────────────────────────────────

print('\n=== Resolve: GGUF LLM (q4_0) ===')

def test_resolve_gguf():
    files = resolver.resolve_files(PKG_GEMMA3, manifest='q4_0')
    gguf_files = [p for vp, p in files.items() if vp.endswith('.gguf')]
    assert_true(len(gguf_files) >= 1, 'no gguf files')
    for gf in gguf_files:
        assert_true(os.path.exists(gf), f'missing: {gf}')
        assert_true(os.path.getsize(gf) > 0, f'empty: {gf}')

test('Resolve GGUF files', test_resolve_gguf)

# ─── Test 6: resolve_gguf convenience ───────────────────────────────────

def test_resolve_gguf_convenience():
    paths = resolve_gguf(PKG_GEMMA3, manifest='q4_0', cache_dir=CACHE_DIR)
    assert_true(len(paths) >= 1, 'no paths')
    for p in paths:
        assert_true(p.endswith('.gguf'), f'not gguf: {p}')
        assert_true(os.path.exists(p), f'missing: {p}')

test('resolve_gguf() convenience', test_resolve_gguf_convenience)

# ─── Test 7: SHA256 verification ────────────────────────────────────────

print('\n=== SHA256 Verification ===')

def test_sha256_verify():
    # Clear cache and re-resolve with verify
    vr = ModelResolver(cache_dir=CACHE_DIR + '-sha', verify_sha256=True)
    local_dir = vr.resolve(PKG_EMBEDDING, manifest='q4f16')
    # Spot-check config.json
    fm = vr.get_filemap(PKG_EMBEDDING)
    config_entry = fm['files']['config.json']
    fp = os.path.join(local_dir, 'config.json')
    h = hashlib.sha256(open(fp, 'rb').read()).hexdigest()
    assert_eq(h, config_entry['sha256'], 'SHA256 mismatch')
    shutil.rmtree(CACHE_DIR + '-sha')

test('SHA256 verification on resolve', test_sha256_verify)

# ─── Test 8: Cache skip (second resolve is fast) ────────────────────────

print('\n=== Cache Skip ===')

def test_cache_skip():
    t0 = time.time()
    resolver.resolve(PKG_EMBEDDING, manifest='q4f16')
    dt = time.time() - t0
    # Should be <1s since all files cached
    assert_true(dt < 2.0, f'second resolve too slow: {dt:.1f}s')

test('Second resolve skips (cached)', test_cache_skip)

# ─── Test 9: CLI list command ───────────────────────────────────────────

print('\n=== CLI: list command ===')

def test_cli_list():
    import subprocess
    r = subprocess.run(
        [sys.executable, '/home/claude/WebModelDelivery/model_resolver.py',
         'list', PKG_EMBEDDING, '--cache-dir', CACHE_DIR],
        capture_output=True, text=True
    )
    assert_eq(r.returncode, 0, f'exit code {r.returncode}')
    assert_true('q4f16' in r.stdout, f'q4f16 not in output: {r.stdout}')

test('CLI: list manifests', test_cli_list)

# ─── Summary ────────────────────────────────────────────────────────────

print(f'\n{"="*50}')
print(f'Python resolver tests: {passed} passed, {failed} failed')
if failed:
    sys.exit(1)
else:
    print('All tests passed ✓')
