#!/usr/bin/env python3
"""
gguf-meta.py — Extract metadata from GGUF files as JSON.

Used by model-packager.sh to auto-detect model properties (architecture,
quantization, context length, etc.) and generate appropriate manifests.

Usage:
    python3 gguf-meta.py model.gguf
    python3 gguf-meta.py model.gguf mmproj.gguf
    python3 gguf-meta.py --classify model.gguf    # just: "llm" or "mmproj"
    python3 gguf-meta.py --quant model.gguf        # just: "Q4_0" etc.

Output: JSON object with metadata for each file.
"""

import struct
import json
import sys
import os
import re

# ─── GGUF file type enum → quantization name ────────────────────────────────

FILE_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
    13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    19: "IQ2_XXS", 20: "IQ2_XS", 21: "IQ3_XXS",
    24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S",
    27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M",
}

# Architecture names that indicate vision/audio projectors (not standalone LLMs)
MMPROJ_ARCHITECTURES = {"clip", "mllama_vision", "minicpmv", "wavtokenizer-dec"}

# ─── GGUF parser ─────────────────────────────────────────────────────────────

GGUF_VALUE_TYPES = {
    0: ('B', 1),    # UINT8
    1: ('b', 1),    # INT8
    2: ('H', 2),    # UINT16
    3: ('h', 2),    # INT16
    4: ('I', 4),    # UINT32
    5: ('i', 4),    # INT32
    6: ('f', 4),    # FLOAT32
    7: ('?', 1),    # BOOL
    8: None,         # STRING
    9: None,         # ARRAY
    10: ('Q', 8),   # UINT64
    11: ('q', 8),   # INT64
    12: ('d', 8),   # FLOAT64
}


def read_gguf_metadata(path, max_kv=300):
    """Parse GGUF file header and return metadata dict."""
    def read_string(f):
        slen = struct.unpack('<Q', f.read(8))[0]
        return f.read(slen).decode('utf-8', errors='replace')

    def read_value(f, vtype, depth=0):
        if vtype == 8:  # STRING
            return read_string(f)
        if vtype == 9:  # ARRAY
            atype = struct.unpack('<I', f.read(4))[0]
            alen = struct.unpack('<Q', f.read(8))[0]
            # For large arrays (e.g. tokenizer vocab), just return length
            if alen > 100:
                # Skip the array data
                if atype in GGUF_VALUE_TYPES and GGUF_VALUE_TYPES[atype]:
                    fmt, size = GGUF_VALUE_TYPES[atype]
                    f.seek(alen * size, 1)
                elif atype == 8:  # array of strings
                    for _ in range(alen):
                        slen = struct.unpack('<Q', f.read(8))[0]
                        f.seek(slen, 1)
                else:
                    # Can't skip unknown nested types
                    return f"[array of {alen} items, type {atype}]"
                return f"[{alen} items]"
            return [read_value(f, atype, depth + 1) for _ in range(alen)]
        info = GGUF_VALUE_TYPES.get(vtype)
        if info:
            fmt, size = info
            return struct.unpack(f'<{fmt}', f.read(size))[0]
        return None

    try:
        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                return {"error": f"Not a GGUF file (magic: {magic!r})"}

            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            kv_count = struct.unpack('<Q', f.read(8))[0]

            meta = {}
            for _ in range(min(kv_count, max_kv)):
                try:
                    key = read_string(f)
                    vtype = struct.unpack('<I', f.read(4))[0]
                    value = read_value(f, vtype)
                    meta[key] = value
                except Exception:
                    break

            return {
                "gguf_version": version,
                "tensor_count": tensor_count,
                "kv_count": kv_count,
                "metadata": meta,
            }
    except Exception as e:
        return {"error": str(e)}


# ─── Higher-level analysis ───────────────────────────────────────────────────

def detect_quant_from_filename(filename):
    """Extract quantization type from GGUF filename."""
    basename = os.path.basename(filename).lower()
    patterns = [
        r'[_\-\.]((?:iq|q|f)\d+(?:_[a-zA-Z0-9]+)*)',
        r'[_\-\.](fp16|fp32|bf16)',
    ]
    for pat in patterns:
        m = re.search(pat, basename)
        if m:
            raw = m.group(1).upper()
            # Normalize: FP16 → F16
            raw = raw.replace('FP', 'F')
            return raw
    return None


def detect_quant_from_metadata(meta):
    """Get quantization from general.file_type metadata field."""
    ft = meta.get("metadata", {}).get("general.file_type")
    if ft is not None and isinstance(ft, int):
        return FILE_TYPE_NAMES.get(ft)
    return None


def classify_gguf(filename, parsed):
    """Classify a GGUF as 'llm' or 'mmproj'."""
    # Check filename first (most reliable)
    if 'mmproj' in os.path.basename(filename).lower():
        return 'mmproj'

    # Check architecture from metadata
    arch = parsed.get("metadata", {}).get("general.architecture", "")
    if arch.lower() in MMPROJ_ARCHITECTURES:
        return 'mmproj'

    return 'llm'


def analyze_gguf(filepath):
    """Full analysis of a single GGUF file."""
    parsed = read_gguf_metadata(filepath)
    if "error" in parsed:
        return parsed

    meta = parsed.get("metadata", {})
    arch = meta.get("general.architecture", "unknown")
    classification = classify_gguf(filepath, parsed)

    # Quantization: try metadata first, fall back to filename
    quant = detect_quant_from_metadata(parsed)
    if not quant:
        quant = detect_quant_from_filename(filepath)
    if not quant:
        quant = "unknown"

    # Architecture-specific parameters
    params = {}
    arch_prefix = arch + "."
    for key, val in meta.items():
        if key.startswith(arch_prefix):
            short_key = key[len(arch_prefix):]
            # Only include scalar values, not large arrays
            if isinstance(val, (int, float, bool, str)) and not str(val).startswith('['):
                params[short_key] = val

    # Common fields across architectures
    result = {
        "file": os.path.basename(filepath),
        "file_size": os.path.getsize(filepath),
        "classification": classification,
        "architecture": arch,
        "quantization": quant,
        "tensor_count": parsed["tensor_count"],
        "gguf_version": parsed["gguf_version"],
    }

    # Add well-known parameters if present
    known_fields = {
        "context_length": params.get("context_length"),
        "embedding_length": params.get("embedding_length"),
        "block_count": params.get("block_count"),
        "feed_forward_length": params.get("feed_forward_length"),
        "head_count": params.get("attention.head_count"),
        "head_count_kv": params.get("attention.head_count_kv"),
        "vocab_size": None,  # filled below
    }

    # Vocab size from tokenizer
    tokens = meta.get("tokenizer.ggml.tokens")
    if isinstance(tokens, str) and tokens.startswith('['):
        # Was truncated to "[N items]"
        m = re.match(r'\[(\d+) items\]', tokens)
        if m:
            known_fields["vocab_size"] = int(m.group(1))
    elif isinstance(tokens, list):
        known_fields["vocab_size"] = len(tokens)

    # General name
    name = meta.get("general.name")
    if name:
        result["name"] = name

    # Add non-None known fields
    for k, v in known_fields.items():
        if v is not None:
            result[k] = v

    # Add full architecture params for reference
    if params:
        result["arch_params"] = params

    return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not args or args[0] in ('-h', '--help'):
        print(__doc__.strip())
        sys.exit(0)

    mode = 'full'
    if args[0] == '--classify':
        mode = 'classify'
        args = args[1:]
    elif args[0] == '--quant':
        mode = 'quant'
        args = args[1:]
    elif args[0] == '--json':
        mode = 'full'
        args = args[1:]

    if not args:
        print("Error: no GGUF files specified", file=sys.stderr)
        sys.exit(1)

    results = {}
    for filepath in args:
        if not os.path.isfile(filepath):
            results[filepath] = {"error": f"File not found: {filepath}"}
            continue

        analysis = analyze_gguf(filepath)
        basename = os.path.basename(filepath)
        results[basename] = analysis

    if mode == 'classify':
        for name, info in results.items():
            print(info.get("classification", "unknown"))
    elif mode == 'quant':
        for name, info in results.items():
            print(info.get("quantization", "unknown"))
    else:
        json.dump(results, sys.stdout, indent=2, default=str)
        print()


if __name__ == '__main__':
    main()
