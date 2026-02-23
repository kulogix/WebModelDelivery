#!/bin/bash
set -euo pipefail

# =============================================================================
# setup-harness.sh — Set up the WebModelDelivery test harness
#
# Creates a harness directory with all required files:
#   - Transformers.js bundle       (ONNX mode)
#   - ONNX Runtime WASM files      (ONNX mode)
#   - wllama WASM + JS bundle      (LLM mode)
#   - Alpine.js (UI)               (ONNX mode)
#   - model-sw.js (Service Worker)
#   - serve.mjs (dev server)
#   - run-test.mjs / run-test-llm.mjs (headless test runners)
#
# Usage:
#   ./setup-harness.sh                         # Creates ONNX harness in ./harness/
#   ./setup-harness.sh --llm                   # Creates LLM harness in ./harness-llm/
#   ./setup-harness.sh --llm ./my-llm-dir      # LLM harness in custom directory
#   ./setup-harness.sh ./my-test-dir            # ONNX harness in custom directory
#   ./setup-harness.sh --with-models            # Also package sample models (if available)
# =============================================================================

LLM_MODE=false
WITH_MODELS=false
POSITIONAL_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --llm)         LLM_MODE=true ;;
    --with-models) WITH_MODELS=true ;;
    --help|-h)
      echo "Usage: $0 [--llm] [HARNESS_DIR] [--with-models]"
      echo ""
      echo "Sets up a test harness directory with all dependencies."
      echo ""
      echo "Modes:"
      echo "  (default)   ONNX/Transformers.js harness (default dir: ./harness)"
      echo "  --llm       wllama/GGUF LLM harness (default dir: ./harness-llm)"
      echo ""
      echo "Options:"
      echo "  --with-models   Prompt for model packaging after setup"
      exit 0
      ;;
    -*) echo "Unknown option: $arg" >&2; exit 1 ;;
    *)  POSITIONAL_ARGS+=("$arg") ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if $LLM_MODE; then
  HARNESS_DIR="${POSITIONAL_ARGS[0]:-./harness-llm}"
else
  HARNESS_DIR="${POSITIONAL_ARGS[0]:-./harness}"
fi

echo "=== WebModelDelivery Harness Setup ==="
echo "Mode:   $($LLM_MODE && echo 'LLM (wllama/GGUF)' || echo 'ONNX (Transformers.js)')"
echo "Output: $HARNESS_DIR"
echo ""

# ─── Create directory ───────────────────────────────────────────────────────
mkdir -p "$HARNESS_DIR"

# ─── Install npm packages ──────────────────────────────────────────────────
echo "--- Installing npm packages ---"
cd "$HARNESS_DIR"

if [ ! -f package.json ]; then
  npm init -y --quiet 2>/dev/null
fi

if $LLM_MODE; then
  npm install --save @wllama/wllama 2>/dev/null
else
  npm install --save @huggingface/transformers 2>/dev/null
  npm install --save onnxruntime-web 2>/dev/null
fi
npm install --save-dev puppeteer-core 2>/dev/null

echo ""

# ─── Copy runtime files ────────────────────────────────────────────────────
echo "--- Copying runtime files ---"

if $LLM_MODE; then
  # ── wllama files ──
  mkdir -p single-thread multi-thread

  WLLAMA_ESM="node_modules/@wllama/wllama/esm"
  if [ -d "$WLLAMA_ESM" ]; then
    cp "$WLLAMA_ESM/index.js" wllama.js
    echo "  ✓ wllama.js"

    if [ -f "$WLLAMA_ESM/single-thread/wllama.wasm" ]; then
      cp "$WLLAMA_ESM/single-thread/wllama.wasm" single-thread/
      echo "  ✓ single-thread/wllama.wasm"
    else
      echo "  ✗ single-thread/wllama.wasm NOT FOUND"
    fi

    if [ -f "$WLLAMA_ESM/multi-thread/wllama.wasm" ]; then
      cp "$WLLAMA_ESM/multi-thread/wllama.wasm" multi-thread/
      echo "  ✓ multi-thread/wllama.wasm"
    else
      echo "  ⚠ multi-thread/wllama.wasm not found (optional, single-thread still works)"
    fi

    # Copy workers-code if present
    if [ -d "$WLLAMA_ESM/workers-code" ]; then
      cp -r "$WLLAMA_ESM/workers-code" .
      echo "  ✓ workers-code/"
    fi

    # Copy glue if present
    if [ -d "$WLLAMA_ESM/glue" ]; then
      cp -r "$WLLAMA_ESM/glue" .
      echo "  ✓ glue/"
    fi
  else
    echo "  ✗ wllama ESM bundle NOT FOUND in $WLLAMA_ESM"
  fi

else
  # ── ONNX / Transformers.js files ──
  if [ -f node_modules/@huggingface/transformers/dist/transformers.js ]; then
    cp node_modules/@huggingface/transformers/dist/transformers.js .
    echo "  ✓ transformers.js"
  else
    echo "  ✗ transformers.js NOT FOUND"
  fi

  ORT_DIR="node_modules/onnxruntime-web/dist"
  if [ -d "$ORT_DIR" ]; then
    for f in ort-wasm-simd-threaded.jsep.wasm ort-wasm-simd-threaded.jsep.mjs \
             ort-wasm-simd-threaded.wasm ort-wasm-simd-threaded.mjs; do
      if [ -f "$ORT_DIR/$f" ]; then
        cp "$ORT_DIR/$f" .
        echo "  ✓ $f"
      fi
    done

    for f in ort.mjs ort.min.mjs ort.wasm.mjs ort.wasm.min.mjs \
             ort.all.mjs ort.all.min.mjs ort.all.bundle.min.mjs \
             ort.bundle.min.mjs ort.webgpu.mjs ort.webgpu.min.mjs \
             ort.webgpu.bundle.min.mjs ort.webgl.mjs ort.webgl.min.mjs \
             ort.node.min.mjs ort.wasm.bundle.min.mjs; do
      if [ -f "$ORT_DIR/$f" ]; then
        cp "$ORT_DIR/$f" .
      fi
    done
    echo "  ✓ ORT bundles"
  else
    echo "  ✗ onnxruntime-web dist NOT FOUND"
  fi
fi

# ─── Copy project files ────────────────────────────────────────────────────
echo ""
echo "--- Copying WebModelDelivery files ---"

# Always copy model-sw.js and serve.mjs
for f in model-sw.js serve.mjs; do
  if [ -f "$SCRIPT_DIR/$f" ]; then
    cp "$SCRIPT_DIR/$f" .
    echo "  ✓ $f"
  else
    echo "  ✗ $f NOT FOUND in $SCRIPT_DIR"
  fi
done

if $LLM_MODE; then
  # Copy LLM-specific files
  for f in run-test-llm.mjs harness-llm-index.html; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
      target="$f"
      [[ "$f" == "harness-llm-index.html" ]] && target="index.html"
      cp "$SCRIPT_DIR/$f" "$target"
      echo "  ✓ $target (from $f)"
    else
      echo "  ✗ $f NOT FOUND in $SCRIPT_DIR"
    fi
  done
else
  # Copy ONNX-specific files
  for f in run-test.mjs harness-index.html; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
      target="$f"
      [[ "$f" == "harness-index.html" ]] && target="index.html"
      cp "$SCRIPT_DIR/$f" "$target"
      echo "  ✓ $target (from $f)"
    else
      echo "  ✗ $f NOT FOUND in $SCRIPT_DIR"
    fi
  done

  # Alpine.js
  if [ ! -f alpine.min.js ]; then
    echo "  Downloading Alpine.js..."
    if curl -sLo alpine.min.js https://cdn.jsdelivr.net/npm/alpinejs@3/dist/cdn.min.js 2>/dev/null && [ -s alpine.min.js ]; then
      echo "  ✓ alpine.min.js (from CDN)"
    else
      # Fallback: install via npm
      rm -f alpine.min.js
      npm install --save alpinejs 2>/dev/null
      if [ -f node_modules/alpinejs/dist/cdn.min.js ]; then
        cp node_modules/alpinejs/dist/cdn.min.js alpine.min.js
        echo "  ✓ alpine.min.js (from npm)"
      else
        echo "  ✗ Alpine.js not available (harness UI won't render)"
      fi
    fi
  fi
fi

# ─── Check for Chrome ──────────────────────────────────────────────────────
echo ""
echo "--- Chrome/Chromium ---"
CHROME_FOUND=false
for p in /opt/google/chrome/chrome /usr/bin/google-chrome /usr/bin/chromium-browser \
         /usr/bin/chromium /snap/bin/chromium \
         "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"; do
  if [ -x "$p" ] 2>/dev/null; then
    echo "  ✓ Found: $p"
    CHROME_FOUND=true
    break
  fi
done

if [ "$CHROME_FOUND" = false ]; then
  echo "  ⚠ Chrome/Chromium not found."
  echo "    For headless tests, install Chrome or set CHROME_PATH."
  echo "    Interactive testing works without Chrome (just use the browser)."
fi

# ─── Package models (optional) ─────────────────────────────────────────────
if [ "$WITH_MODELS" = true ] && [ -x "$SCRIPT_DIR/model-packager.sh" ]; then
  echo ""
  echo "--- Packaging sample models ---"
  if $LLM_MODE; then
    echo "  To package a GGUF model:"
    echo "    $SCRIPT_DIR/model-packager.sh -o $HARNESS_DIR/pkg-mymodel \\"
    echo "      --gguf-split /path/to/llama-gguf-split /path/to/model.gguf"
  else
    echo "  To package an ONNX model:"
    echo "    $SCRIPT_DIR/model-packager.sh -o $HARNESS_DIR/pkg-embedding /path/to/model/"
  fi
fi

# ─── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo ""

if $LLM_MODE; then
  echo "  1. Package your GGUF model:"
  echo "     $SCRIPT_DIR/model-packager.sh -o $HARNESS_DIR/pkg-gemma3 \\"
  echo "       --gguf-split /path/to/llama-gguf-split /path/to/model-q4_0.gguf"
  echo ""
  echo "  2. Run interactively:"
  echo "     cd $HARNESS_DIR && node serve.mjs"
  echo "     → Open http://localhost:PORT in your browser"
  echo ""
  echo "  3. Run headless tests:"
  echo "     cd $HARNESS_DIR && node run-test-llm.mjs"
  echo "     cd $HARNESS_DIR && node run-test-llm.mjs --skip-inference  # quick"
else
  echo "  1. Package your models:"
  echo "     $SCRIPT_DIR/model-packager.sh -o $HARNESS_DIR/pkg-embedding /path/to/model/"
  echo ""
  echo "  2. Run interactively:"
  echo "     cd $HARNESS_DIR && node serve.mjs"
  echo "     → Open http://localhost:PORT in your browser"
  echo ""
  echo "  3. Run headless tests:"
  echo "     cd $HARNESS_DIR && node run-test.mjs"
  echo "     cd $HARNESS_DIR && node run-test.mjs --skip-inference  # quick"
fi
echo ""
