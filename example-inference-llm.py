#!/usr/bin/env python3
"""
example-inference-llm.py — End-to-end GGUF LLM inference via model_resolver

Demonstrates:
  1. Resolving a GGUF model from a local flat-repo (or CDN) using model_resolver
  2. Loading with llama-cpp-python (pip install llama-cpp-python)
  3. Single-turn and multi-turn chat completion

Usage:
  python example-inference-llm.py --source /path/to/pkg-gemma3

  # CDN source:
  python example-inference-llm.py \
    --source https://cdn.jsdelivr.net/gh/user/cdn-llm@v1 \
    --manifest q4_0

Requirements:
  pip install llama-cpp-python
"""

import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_resolver import ModelResolver, resolve_gguf

try:
    from llama_cpp import Llama
except ImportError:
    sys.exit("pip install llama-cpp-python")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True,
                   help='Local flat-repo path or CDN URL for GGUF model')
    p.add_argument('--manifest', default='q4_0')
    p.add_argument('--cache-dir', default='./.model-cache')
    p.add_argument('--max-tokens', type=int, default=128)
    args = p.parse_args()

    prog = lambda p: print(f"\r  [{p['percent']:3d}%] {p['file'][:50]:<50s}", end='', flush=True)

    # ── 1. Resolve GGUF model ────────────────────────────────────────────
    print("=" * 60)
    print("LLM INFERENCE via model_resolver + llama-cpp-python")
    print("=" * 60)

    print(f"\nResolving GGUF model from: {args.source}")
    t0 = time.time()
    gguf_paths = resolve_gguf(args.source, manifest=args.manifest,
                              cache_dir=args.cache_dir, on_progress=prog)
    model_path = gguf_paths[0]
    print(f"\n  → {model_path}  ({time.time()-t0:.1f}s)")
    print(f"  Size: {os.path.getsize(model_path)/1048576:.1f} MB")

    # ── 2. Load model ────────────────────────────────────────────────────
    print("\nLoading model with llama-cpp-python...")
    t0 = time.time()
    llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=0, verbose=False)
    print(f"  ✓ Model loaded ({time.time()-t0:.1f}s)")

    # ── 3. Single-turn inference ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SINGLE-TURN INFERENCE")
    print("-" * 60)

    prompts = [
        "What is 2 + 2? Answer in one sentence.",
        "Explain gravity to a 5 year old in two sentences.",
        "Write a haiku about programming.",
    ]

    for prompt in prompts:
        print(f"\n  User: {prompt}")
        t0 = time.time()
        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=args.max_tokens, temperature=0.7,
        )
        response = result['choices'][0]['message']['content']
        dt = time.time() - t0
        usage = result.get('usage', {})
        print(f"  Model: {response.strip()}")
        print(f"  ({dt:.1f}s, {usage.get('completion_tokens', '?')} tokens)")

    # ── 4. Multi-turn conversation ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("MULTI-TURN CONVERSATION")
    print("-" * 60)

    messages = []
    turns = [
        "What is the capital of France?",
        "What is its population?",
        "And what is a famous landmark there?",
    ]

    for turn in turns:
        messages.append({"role": "user", "content": turn})
        print(f"\n  User: {turn}")
        result = llm.create_chat_completion(
            messages=messages, max_tokens=64, temperature=0.3,
        )
        reply = result['choices'][0]['message']['content'].strip()
        messages.append({"role": "assistant", "content": reply})
        print(f"  Model: {reply}")

    print("\n" + "=" * 60)
    print("✓ All LLM inference complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
