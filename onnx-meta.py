#!/usr/bin/env python3
"""
onnx-meta.py — Extract metadata from ONNX model repositories as JSON.

Used by model-packager.sh to auto-detect model properties (task type,
architecture, embedding dimensions, etc.) and enrich filemap.json.

Inspects config.json and other metadata files (modules.json,
sentence_bert_config.json, tokenizer_config.json, generation_config.json)
in the model directory to classify the model and extract useful metadata.

Usage:
    python3 onnx-meta.py /path/to/model/dir
    python3 onnx-meta.py --classify /path/to/model/dir    # just task type
    python3 onnx-meta.py --json /path/to/model/dir         # full JSON

Output: JSON object with model metadata including classification.
"""

import json
import sys
import os
import re

# ─── Architecture suffix → task type mapping ─────────────────────────────────
# Tier 1: High-confidence direct mappings from architecture class name suffixes.
# These suffixes are standardized across the HuggingFace transformers library.
# The key is checked with str.endswith() against architectures[0].

ARCHITECTURE_TASK_MAP = {
    # Text generation
    'ForCausalLM':                    'text-generation',
    'LMHeadModel':                    'text-generation',       # GPT2LMHeadModel etc.

    # Seq2seq generation (translation, summarization)
    'ForSeq2SeqLM':                   'text-generation-seq2seq',
    'ForConditionalGeneration':       'text-generation-seq2seq',  # T5, BART, etc.

    # Classification (needs disambiguation → reranker vs classifier vs zero-shot)
    'ForSequenceClassification':      'sequence-classification',  # disambiguated later

    # Token classification (NER, POS tagging)
    'ForTokenClassification':         'token-classification',

    # Extractive QA
    'ForQuestionAnswering':           'question-answering',

    # Fill-mask (MLM)
    'ForMaskedLM':                    'fill-mask',
    'ForPreTraining':                 'fill-mask',

    # Multiple choice
    'ForMultipleChoice':              'multiple-choice',

    # Vision
    'ForImageClassification':         'image-classification',
    'ForObjectDetection':             'object-detection',
    'ForImageSegmentation':           'image-segmentation',
    'ForSemanticSegmentation':        'image-segmentation',
    'ForDepthEstimation':             'depth-estimation',
    'ForVideoClassification':         'video-classification',
    'ForMaskedImageModeling':         'masked-image-modeling',
    'ForImageTextToText':             'image-text-to-text',

    # Audio
    'ForAudioClassification':         'audio-classification',
    'ForCTC':                         'speech-recognition',
    'ForSpeechSeq2Seq':               'speech-recognition',
    'ForAudioFrameClassification':    'audio-frame-classification',
    'ForAudioXVector':                'audio-xvector',

    # Multimodal
    'ForMultimodalLM':                'multimodal-generation',
    'ForVision2Seq':                  'image-to-text',
    'ForZeroShotObjectDetection':     'zero-shot-object-detection',
    'ForZeroShotImageClassification': 'zero-shot-image-classification',
}

# ─── Known reranker / cross-encoder name patterns ────────────────────────────

RERANKER_NAME_PATTERNS = [
    r'rerank', r'cross[\-_]?encoder', r'ms[\-_]?marco',
    r'bge[\-_]reranker', r'jina[\-_]reranker', r'mxbai[\-_]rerank',
    r'cohere[\-_]rerank',
]

# ─── Known embedding model name patterns ─────────────────────────────────────

EMBEDDING_NAME_PATTERNS = [
    r'embed', r'e5[\-_]', r'gte[\-_]', r'bge[\-_](?!reranker)',
    r'sentence[\-_]?transformer', r'all[\-_]Mini', r'all[\-_]mpnet',
    r'nomic[\-_]embed', r'jina[\-_]embed', r'instructor',
    r'arctic[\-_]embed', r'stella', r'mxbai[\-_]embed',
]

# ─── id2label patterns for zero-shot NLI models ─────────────────────────────

NLI_LABELS = {'entailment', 'contradiction', 'neutral',
              'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'}


def is_generic_label(label):
    """Check if a label is a generic/default HuggingFace label (not meaningful)."""
    if not label:
        return True
    # Patterns: LABEL_0, LABEL_1, Label_0, label_0, etc.
    if re.match(r'^[Ll][Aa][Bb][Ee][Ll][_\-]?\d+$', label):
        return True
    return False


def classify_sequence_model(config, file_list, model_name):
    """
    Disambiguate a ForSequenceClassification model into:
      - 'reranker'              (cross-encoder for relevance scoring)
      - 'zero-shot-classification' (NLI-based)
      - 'text-classification'   (sentiment, topic, etc.)
    """
    id2label = config.get('id2label', {})
    num_labels = config.get('num_labels', len(id2label) if id2label else 0)
    label_values = set(str(v) for v in id2label.values()) if id2label else set()

    # ── Check for zero-shot NLI model ──
    if label_values & NLI_LABELS:
        return 'zero-shot-classification'

    # ── Check for reranker by label pattern ──
    # Rerankers typically have 1 label or only generic labels
    all_generic = all(is_generic_label(str(v)) for v in id2label.values()) if id2label else True

    if all_generic and num_labels <= 1:
        return 'reranker'

    # ── Check model name for reranker indicators ──
    name_lower = model_name.lower()
    for pat in RERANKER_NAME_PATTERNS:
        if re.search(pat, name_lower):
            return 'reranker'

    # ── Check model name for reranker in _name_or_path ──
    name_or_path = config.get('_name_or_path', '')
    for pat in RERANKER_NAME_PATTERNS:
        if re.search(pat, name_or_path.lower()):
            return 'reranker'

    # ── Default: if all labels are generic, lean toward reranker ──
    if all_generic:
        return 'reranker'

    # ── Otherwise it's a real classifier ──
    return 'text-classification'


def classify_base_model(config, file_list, model_name):
    """
    Classify a base model (no For* task head) as:
      - 'embedding'           (sentence embedding / bi-encoder)
      - 'feature-extraction'  (generic backbone)
    """
    # ── Strong signals for embedding model ──

    # sentence-transformers ecosystem files
    st_files = {'modules.json', 'sentence_bert_config.json',
                'config_sentence_transformers.json'}
    if st_files & set(file_list):
        return 'embedding'

    # Pooling directory (sentence-transformers convention)
    if any(f.startswith('1_Pooling/') or f == '1_Pooling' for f in file_list):
        return 'embedding'

    # Bidirectional attention flag (e.g. embeddinggemma)
    if config.get('use_bidirectional_attention'):
        return 'embedding'

    # ── Name-based heuristics ──
    name_lower = model_name.lower()
    name_or_path = config.get('_name_or_path', '').lower()
    combined = name_lower + ' ' + name_or_path

    for pat in EMBEDDING_NAME_PATTERNS:
        if re.search(pat, combined):
            return 'embedding'

    # ── Default: feature extraction (generic backbone) ──
    return 'feature-extraction'


def detect_task_type(config, file_list, model_name):
    """
    Main classification entry point.
    Returns a task type string.
    """
    architectures = config.get('architectures', [])
    if not architectures:
        # No architecture info → use heuristics
        return classify_base_model(config, file_list, model_name)

    arch = architectures[0]

    # ── Check Tier 1: architecture suffix map ──
    for suffix, task in ARCHITECTURE_TASK_MAP.items():
        if arch.endswith(suffix):
            if task == 'sequence-classification':
                return classify_sequence_model(config, file_list, model_name)
            return task

    # ── No suffix matched → base model ──
    return classify_base_model(config, file_list, model_name)


# ─── Transformers.js AutoModel class recommendation ─────────────────────────

TASK_TO_AUTOMODEL = {
    'embedding':                    'AutoModel',
    'feature-extraction':           'AutoModel',
    'text-generation':              'AutoModelForCausalLM',
    'text-generation-seq2seq':      'AutoModelForSeq2SeqLM',
    'reranker':                     'AutoModelForSequenceClassification',
    'text-classification':          'AutoModelForSequenceClassification',
    'zero-shot-classification':     'AutoModelForSequenceClassification',
    'token-classification':         'AutoModelForTokenClassification',
    'question-answering':           'AutoModelForQuestionAnswering',
    'fill-mask':                    'AutoModelForMaskedLM',
    'image-classification':         'AutoModelForImageClassification',
    'object-detection':             'AutoModelForObjectDetection',
    'image-segmentation':           'AutoModelForImageSegmentation',
    'depth-estimation':             'AutoModelForDepthEstimation',
    'speech-recognition':           'AutoModelForCTC',
    'audio-classification':         'AutoModelForAudioClassification',
    'image-to-text':                'AutoModelForVision2Seq',
    'image-text-to-text':           'AutoModelForImageTextToText',
    'multimodal-generation':        'AutoModelForMultimodalLM',
}


def extract_onnx_variants(file_list):
    """
    Find ONNX model files and their quantization variants.
    Returns list of {name, path, has_external_data} dicts.
    """
    variants = []
    onnx_files = sorted(f for f in file_list if f.endswith('.onnx'))
    onnx_data_files = set(f for f in file_list if f.endswith('.onnx_data'))

    for onnx_file in onnx_files:
        basename = os.path.basename(onnx_file)
        stem = re.sub(r'\.onnx$', '', basename)
        # Derive variant name: model_q4f16.onnx → q4f16, model.onnx → default
        name = re.sub(r'^model_?', '', stem) or 'default'
        has_data = (onnx_file + '_data') in onnx_data_files
        variants.append({
            'name': name,
            'file': onnx_file,
            'has_external_data': has_data,
        })
    return variants


def analyze_model_dir(model_dir):
    """Full analysis of a model directory."""
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.isfile(config_path):
        return {"error": f"No config.json found in {model_dir}"}

    try:
        with open(config_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return {"error": f"Failed to read config.json: {e}"}

    # ── Gather file list (for heuristics) ──
    file_list = []
    for root, dirs, files in os.walk(model_dir):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), model_dir)
            file_list.append(rel)
    # Also check just immediate directory for flat repos
    dir_name = os.path.basename(os.path.abspath(model_dir))

    # ── Classify ──
    task_type = detect_task_type(config, file_list, dir_name)

    # ── Build result ──
    result = {
        'classification': task_type,
        'auto_model_class': TASK_TO_AUTOMODEL.get(task_type, 'AutoModel'),
    }

    # Architecture info
    archs = config.get('architectures', [])
    if archs:
        result['architecture'] = archs[0]

    result['model_type'] = config.get('model_type', 'unknown')

    # Name from config
    name = config.get('_name_or_path') or config.get('name')
    if name:
        result['name'] = name

    # ── Dimensionality info ──
    hidden_size = config.get('hidden_size')
    if hidden_size:
        result['hidden_size'] = hidden_size

    # For embedding models, hidden_size is the embedding dimension
    if task_type in ('embedding', 'feature-extraction') and hidden_size:
        result['embedding_dimension'] = hidden_size

    # ── Model structure ──
    for key in ['num_hidden_layers', 'num_attention_heads',
                'num_key_value_heads', 'intermediate_size',
                'vocab_size', 'max_position_embeddings',
                'head_dim']:
        val = config.get(key)
        if val is not None:
            result[key] = val

    # ── Classification-specific ──
    id2label = config.get('id2label')
    if id2label:
        result['num_labels'] = len(id2label)
        result['labels'] = list(id2label.values())

    problem_type = config.get('problem_type')
    if problem_type:
        result['problem_type'] = problem_type

    # ── Bidirectional attention (embedding indicator) ──
    if config.get('use_bidirectional_attention'):
        result['use_bidirectional_attention'] = True

    # ── Transformers.js config ──
    tj_config = config.get('transformers.js_config')
    if tj_config:
        result['transformers_js_config'] = tj_config

    # ── ONNX variants ──
    variants = extract_onnx_variants(file_list)
    if variants:
        result['onnx_variants'] = variants

    # ── Sentence-transformers metadata ──
    modules_path = os.path.join(model_dir, 'modules.json')
    if os.path.isfile(modules_path):
        try:
            with open(modules_path) as f:
                modules = json.load(f)
            result['sentence_transformers_modules'] = [
                {'name': m.get('name', ''), 'type': m.get('type', '')}
                for m in modules
            ]
        except Exception:
            pass

    st_config_path = os.path.join(model_dir, 'config_sentence_transformers.json')
    if os.path.isfile(st_config_path):
        try:
            with open(st_config_path) as f:
                st_config = json.load(f)
            if 'prompts' in st_config:
                result['prompts'] = st_config['prompts']
            if 'default_prompt_name' in st_config:
                result['default_prompt_name'] = st_config['default_prompt_name']
        except Exception:
            pass

    # ── Pooling config (sentence-transformers) ──
    pooling_path = os.path.join(model_dir, '1_Pooling', 'config.json')
    if os.path.isfile(pooling_path):
        try:
            with open(pooling_path) as f:
                pool_config = json.load(f)
            # Extract just the pooling mode flags
            pooling_modes = {k: v for k, v in pool_config.items()
                           if k.startswith('pooling_mode_') and v}
            if pooling_modes:
                result['pooling'] = pooling_modes
        except Exception:
            pass

    # ── Generation config ──
    gen_path = os.path.join(model_dir, 'generation_config.json')
    if os.path.isfile(gen_path):
        result['has_generation_config'] = True
        try:
            with open(gen_path) as f:
                gen = json.load(f)
            for key in ['max_new_tokens', 'max_length', 'do_sample',
                        'temperature', 'top_p', 'top_k']:
                if key in gen:
                    result.setdefault('generation_defaults', {})[key] = gen[key]
        except Exception:
            pass

    # ── Tokenizer info ──
    tok_config_path = os.path.join(model_dir, 'tokenizer_config.json')
    if os.path.isfile(tok_config_path):
        try:
            with open(tok_config_path) as f:
                tok = json.load(f)
            result['tokenizer_class'] = tok.get('tokenizer_class', tok.get('model_type', 'unknown'))
            if 'model_max_length' in tok:
                result['model_max_length'] = tok['model_max_length']
        except Exception:
            pass

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
    elif args[0] == '--json':
        mode = 'full'
        args = args[1:]

    if not args:
        print("Error: no model directory specified", file=sys.stderr)
        sys.exit(1)

    model_dir = args[0]
    if not os.path.isdir(model_dir):
        print(json.dumps({"error": f"Not a directory: {model_dir}"}))
        sys.exit(1)

    result = analyze_model_dir(model_dir)

    if mode == 'classify':
        print(result.get('classification', 'unknown'))
    else:
        json.dump(result, sys.stdout, indent=2, default=str)
        print()


if __name__ == '__main__':
    main()
