#!/usr/bin/env python3
"""
example-inference-onnx.py — End-to-end ONNX inference via model_resolver

Demonstrates:
  1. Resolving ONNX models from local flat-repo (or CDN) using model_resolver
  2. Running embedding inference (embeddinggemma-300m q4f16)
  3. Running reranker inference (mxbai-rerank-xsmall quantized)
  4. Computing cosine similarity between embeddings
  5. Reranking search results by relevance

Usage:
  # Local flat repos (no network):
  python example-inference-onnx.py \
    --embedding-source /path/to/pkg-embedding \
    --reranker-source /path/to/pkg-reranker

  # CDN sources:
  python example-inference-onnx.py \
    --embedding-source https://cdn.jsdelivr.net/gh/user/cdn-embedding@v1 \
    --reranker-source https://cdn.jsdelivr.net/gh/user/cdn-reranker@v1

  # Custom cache dir:
  python example-inference-onnx.py --cache-dir ./my-cache ...
"""

import argparse
import json
import math
import os
import sys
import time
import numpy as np

# ─── Dependencies ────────────────────────────────────────────────────────
try:
    import onnxruntime as ort
except ImportError:
    sys.exit("pip install onnxruntime  (or onnxruntime-gpu)")

try:
    from tokenizers import Tokenizer
except ImportError:
    sys.exit("pip install tokenizers")

# Adjust path if running from the WebModelDelivery directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_resolver import ModelResolver


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def main():
    parser = argparse.ArgumentParser(description='ONNX inference via model_resolver')
    parser.add_argument('--embedding-source', required=True,
                        help='Local flat-repo path or CDN URL for embedding model')
    parser.add_argument('--embedding-manifest', default='q4f16',
                        help='Manifest name for embedding model (default: q4f16)')
    parser.add_argument('--reranker-source', required=True,
                        help='Local flat-repo path or CDN URL for reranker model')
    parser.add_argument('--reranker-manifest', default='quantized',
                        help='Manifest name for reranker model (default: quantized)')
    parser.add_argument('--cache-dir', default='./.model-cache',
                        help='Cache directory (default: ./.model-cache)')
    args = parser.parse_args()

    resolver = ModelResolver(cache_dir=args.cache_dir)

    # ─── 1. Resolve embedding model ─────────────────────────────────────
    print("="*60)
    print("Resolving embedding model...")
    t0 = time.time()
    emb_dir = resolver.resolve(
        args.embedding_source,
        manifest=args.embedding_manifest,
        on_progress=lambda p: print(f"\r  [{p['percent']:3d}%] {p['file'][:50]}", end='', flush=True),
    )
    print(f"\n  → {emb_dir}  ({time.time()-t0:.1f}s)")

    # ─── 2. Resolve reranker model ──────────────────────────────────────
    print("\nResolving reranker model...")
    t0 = time.time()
    rr_dir = resolver.resolve(
        args.reranker_source,
        manifest=args.reranker_manifest,
        on_progress=lambda p: print(f"\r  [{p['percent']:3d}%] {p['file'][:50]}", end='', flush=True),
    )
    print(f"\n  → {rr_dir}  ({time.time()-t0:.1f}s)")

    # ─── 3. Load tokenizers ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("Loading tokenizers...")
    emb_tokenizer = Tokenizer.from_file(os.path.join(emb_dir, 'tokenizer.json'))
    rr_tokenizer = Tokenizer.from_file(os.path.join(rr_dir, 'tokenizer.json'))
    print("  ✓ Tokenizers loaded")

    # ─── 4. Create ONNX sessions ────────────────────────────────────────
    print("Creating ONNX sessions...")
    emb_session = ort.InferenceSession(os.path.join(emb_dir, 'onnx', 'model_q4f16.onnx'))
    rr_session = ort.InferenceSession(os.path.join(rr_dir, 'onnx', 'model_quantized.onnx'))
    print(f"  ✓ Embedding: inputs={emb_session.get_inputs()[0].name}, outputs={[o.name for o in emb_session.get_outputs()]}")
    print(f"  ✓ Reranker:  inputs={rr_session.get_inputs()[0].name}, outputs={[o.name for o in rr_session.get_outputs()]}")

    # ─── 5. Embedding inference ─────────────────────────────────────────
    print("\n" + "="*60)
    print("EMBEDDING INFERENCE")
    print("-"*60)

    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps above a sleepy hound.",
        "Machine learning models can run in the browser.",
        "The stock market closed higher on Friday.",
    ]

    embeddings = []
    for sent in sentences:
        encoded = emb_tokenizer.encode(sent)
        ids = np.array([encoded.ids], dtype=np.int64)
        mask = np.array([encoded.attention_mask], dtype=np.int64)

        t0 = time.time()
        outputs = emb_session.run(None, {'input_ids': ids, 'attention_mask': mask})
        dt = time.time() - t0

        # sentence_embedding is the second output
        emb = outputs[1][0] if len(outputs) > 1 else outputs[0][0].mean(axis=0)
        embeddings.append(emb)
        print(f"  \"{sent[:50]}...\"")
        print(f"    → dim={len(emb)}, norm={np.linalg.norm(emb):.4f}, {dt*1000:.0f}ms")

    # Cosine similarities
    print(f"\nCosine similarities:")
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"  [{i}] vs [{j}]: {sim:.4f}  {'← similar!' if sim > 0.7 else ''}")

    # ─── 6. Reranker inference ──────────────────────────────────────────
    print("\n" + "="*60)
    print("RERANKER INFERENCE")
    print("-"*60)

    query = "How do neural networks learn?"
    documents = [
        "Neural networks adjust weights through backpropagation during training.",
        "The stock market experienced volatility this quarter.",
        "Deep learning uses gradient descent to minimize loss functions.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "Transformers use self-attention mechanisms for sequence modeling.",
    ]

    print(f"  Query: \"{query}\"")
    print(f"  Documents: {len(documents)}")

    scores = []
    for doc in documents:
        # Reranker expects query + document as a pair
        # Use [SEP] token between them
        rr_tokenizer.no_padding()
        rr_tokenizer.no_truncation()
        encoded = rr_tokenizer.encode(query, doc)
        ids = np.array([encoded.ids], dtype=np.int64)
        mask = np.array([encoded.attention_mask], dtype=np.int64)

        outputs = rr_session.run(None, {'input_ids': ids, 'attention_mask': mask})
        # Reranker logits — higher = more relevant
        score = float(outputs[0][0][0])
        scores.append(score)

    # Sort by score
    ranked = sorted(enumerate(documents), key=lambda x: scores[x[0]], reverse=True)
    print(f"\n  Ranked results:")
    for rank, (idx, doc) in enumerate(ranked):
        print(f"    #{rank+1} (score: {scores[idx]:+.4f}): \"{doc[:70]}\"")

    print("\n" + "="*60)
    print("✓ All ONNX inference complete")
    print("="*60)


if __name__ == '__main__':
    main()
