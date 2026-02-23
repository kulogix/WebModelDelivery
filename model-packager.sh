#!/usr/bin/env bash
set -euo pipefail

# ── Portable key-value store (replaces bash 4+ associative arrays) ────────
# Uses tab-separated temp files: key\tvalue per line, awk for lookups.
# Works on bash 3.2 (macOS default).
_KV_DIR=""
_kv_setup() {
  _KV_DIR="$(mktemp -d)"
}
_kv_init() { : > "$1"; }                                                      # create/clear
_kv_set()  { printf '%s\t%s\n' "$2" "$3" >> "$1"; }                           # append key\tvalue
_kv_get()  { awk -F'\t' -v k="$2" '$1==k{print $2; exit}' "$1" 2>/dev/null; } # first match
_kv_has()  { awk -F'\t' -v k="$2" 'BEGIN{r=1} $1==k{r=0; exit} END{exit r}' "$1" 2>/dev/null; }
_kv_count(){ awk 'END{print NR}' "$1" 2>/dev/null || echo 0; }                # line count

# =============================================================================
# model-packager.sh v4 — Package model files for JSDelivr CDN delivery
#
# Accepts folders (auto-discover) or individual files. Produces a FLAT output
# directory with a single filemap.json for Service Worker transparent delivery.
#
# Supports MERGE mode: run multiple times against the same output dir to add
# different quantizations. Shared files (same SHA256) are deduplicated — only
# stored once on disk, referenced from all virtual paths.
#
# Usage:
#   # First quantization
#   ./model-packager.sh -o ./cdn-embedding models/embeddinggemma-q4/
#
#   # Add second quantization (shared tokenizer/config reused via SHA256 dedup)
#   ./model-packager.sh -o ./cdn-embedding --merge models/embeddinggemma-q4f16/
#
#   # GGUF with custom shard size
#   ./model-packager.sh -o ./cdn-llm --gguf-shard-size 500M models/gemma.gguf
# =============================================================================

VERSION="4.5.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Defaults ---------------------------------------------------------------
CHUNK_SIZE=19922944          # 19 MiB — safely under JSDelivr's 20 MB limit
GGUF_SPLIT_BIN=""            # empty = search PATH
GGUF_SHARD_SIZE="1800M"      # max gguf-split shard (must be < 2GB for wllama)
KEEP_INTERMEDIATES=false
REMOVE_ORIGINALS=false
OUTPUT_DIR=""
DRY_RUN=false
VERBOSE=false
MERGE=false                  # merge into existing output with dedup
OVERWRITE=false              # wipe existing output first
MANIFEST_NAME=""             # explicit manifest name for this packaging run
INPUTS=()

# Excluded from auto-discovery: only hidden/VCS artifacts
EXCLUDE_PATTERNS=(
  ".git" ".gitattributes" ".gitignore" ".DS_Store" "Thumbs.db"
  ".gitkeep" ".npmignore"
)

# --- Usage ------------------------------------------------------------------
usage() {
  cat <<'EOF'
Usage: model-packager.sh -o OUTPUT_DIR [OPTIONS] INPUT [INPUT...]

Required:
  -o, --output DIR          Target directory (created if needed). Always flat.
  INPUT...                  Folders and/or files.

Options:
  -s, --chunk-size BYTES    Max CDN chunk size in bytes (default: 19922944 = 19 MiB)
  --gguf-split PATH         Path to llama-gguf-split binary (default: search PATH)
  --gguf-shard-size SIZE    Max GGUF shard size, e.g. 1800M (default: 1800M, must < 2G)
  --merge                   Merge into existing output dir. Shared files (same SHA256)
                            are deduplicated — stored once, referenced by all virtual paths.
  --overwrite               Wipe existing output dir before packaging.
  --keep-intermediates      Keep full gguf-split shards (>chunk-size) in output.
  --remove-originals        Delete original inputs after success.
  --exclude PATTERN         Additional glob to exclude in folder discovery (repeatable).
  --dry-run                 Show plan without writing.
  -v, --verbose             Verbose output.
  -h, --help                Show this help.

Multi-quantization workflow:
  # Package q4 model
  ./model-packager.sh -o ./cdn-embedding models/embeddinggemma-q4/

  # Add q4f16 to same repo — tokenizer/config reused (same SHA256)
  ./model-packager.sh -o ./cdn-embedding --merge models/embeddinggemma-q4f16/

  Result: one flat repo, one filemap.json with entries for both quantizations.
  Shared files stored once. Service Worker serves both transparently.

Output (always flat):
  OUTPUT_DIR/
  ├── filemap.json
  ├── config.json                    ← virtual: "config.json" (shared)
  ├── model_q4f16.onnx               ← virtual: "onnx/model_q4f16.onnx"
  ├── model_q4f16.onnx_data.shard.000
  └── ...
EOF
  exit "${1:-0}"
}

# --- Parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output)          OUTPUT_DIR="$2"; shift 2 ;;
    -s|--chunk-size)      CHUNK_SIZE="$2"; shift 2 ;;
    --gguf-split)         GGUF_SPLIT_BIN="$2"; shift 2 ;;
    --gguf-shard-size)    GGUF_SHARD_SIZE="$2"; shift 2 ;;
    --merge)              MERGE=true; shift ;;
    --manifest)           MANIFEST_NAME="$2"; shift 2 ;;
    --overwrite)          OVERWRITE=true; shift ;;
    --keep-intermediates) KEEP_INTERMEDIATES=true; shift ;;
    --remove-originals)   REMOVE_ORIGINALS=true; shift ;;
    --exclude)            EXCLUDE_PATTERNS+=("$2"); shift 2 ;;
    --dry-run)            DRY_RUN=true; shift ;;
    -v|--verbose)         VERBOSE=true; shift ;;
    -h|--help)            usage 0 ;;
    -*)                   echo "Error: Unknown option: $1" >&2; usage 1 ;;
    *)                    INPUTS+=("$1"); shift ;;
  esac
done

# --- Helpers ----------------------------------------------------------------
log()  { echo "  $*"; }
vlog() { $VERBOSE && echo "  [v] $*" || true; }
die()  { echo "Error: $*" >&2; exit 1; }

human_size() {
  local bytes=$1
  if   (( bytes >= 1073741824 )); then printf "%.1f GB" "$(echo "scale=1; $bytes/1073741824" | bc)"
  elif (( bytes >= 1048576 ));    then printf "%.1f MB" "$(echo "scale=1; $bytes/1048576" | bc)"
  elif (( bytes >= 1024 ));       then printf "%.1f KB" "$(echo "scale=1; $bytes/1024" | bc)"
  else printf "%d B" "$bytes"
  fi
}

parse_size() {
  local input="$1" num unit
  if [[ "$input" =~ ^([0-9]+\.?[0-9]*)([GgMmKk]?)$ ]]; then
    num="${BASH_REMATCH[1]}"; unit="${BASH_REMATCH[2]}"
    case "$unit" in
      G|g) echo "$(echo "$num * 1073741824" | bc | cut -d. -f1)" ;;
      M|m) echo "$(echo "$num * 1048576" | bc | cut -d. -f1)" ;;
      K|k) echo "$(echo "$num * 1024" | bc | cut -d. -f1)" ;;
      *) echo "$num" ;;
    esac
  else echo ""; fi
}

file_sha256() {
  if command -v sha256sum &>/dev/null; then
    sha256sum "$1" | cut -d' ' -f1
  elif command -v gsha256sum &>/dev/null; then
    gsha256sum "$1" | cut -d' ' -f1
  else
    # macOS fallback — shasum is pre-installed
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}
get_size()    { stat --format='%s' "$1" 2>/dev/null || stat -f '%z' "$1" 2>/dev/null; }
is_gguf()     { [[ "$1" == *.gguf ]]; }

WLLAMA_MAX=2000000000

needs_gguf_split() {
  local size; size=$(get_size "$1")
  (( size > GGUF_SHARD_SIZE_BYTES )) || (( size > WLLAMA_MAX ))
}

needs_byte_split() {
  local size; size=$(get_size "$1")
  (( size > CHUNK_SIZE ))
}

find_gguf_split() {
  if [[ -n "$GGUF_SPLIT_BIN" ]]; then return 0; fi
  for name in llama-gguf-split gguf-split; do
    if command -v "$name" &>/dev/null; then
      GGUF_SPLIT_BIN="$(command -v "$name")"; return 0
    fi
  done
  return 1
}

should_exclude() {
  local filepath="$1" basename_f
  basename_f="$(basename "$filepath")"
  [[ "$basename_f" == .* ]] && return 0
  [[ "$filepath" == */.* ]] && return 0
  for pat in "${EXCLUDE_PATTERNS[@]}"; do
    # shellcheck disable=SC2254
    case "$basename_f" in $pat) return 0 ;; esac
  done
  return 1
}

get_free_space() {
  # Free space in bytes on the volume containing the given path
  local path="$1"
  # Ensure path exists (use parent if it doesn't yet)
  while [[ ! -e "$path" ]]; do path="$(dirname "$path")"; done
  df -P "$path" | awk 'NR==2 {print $4 * 1024}'
}

# =============================================================================
# Validation
# =============================================================================
[[ -z "$OUTPUT_DIR" ]] && usage 1
[[ ${#INPUTS[@]} -eq 0 ]] && die "no inputs specified"

# Required tools (sha256sum handled by file_sha256 with gsha256sum/shasum fallback)
for cmd in python3 split bc df stat find sort mkdir; do
  command -v "$cmd" &>/dev/null || die "'$cmd' not found — required dependency"
done

# Verify python3 works
python3 -c "import json, sys" 2>/dev/null || die "python3 cannot import json/sys"

# Chunk size — parse human-readable (20M, 19K, etc.) or plain bytes
if ! [[ "$CHUNK_SIZE" =~ ^[0-9]+$ ]]; then
  CHUNK_SIZE="$(parse_size "$CHUNK_SIZE")"
  [[ -z "$CHUNK_SIZE" ]] && die "--chunk-size: cannot parse (use e.g. 20000000, 19M)"
fi
(( CHUNK_SIZE >= 1048576 ))     || die "--chunk-size must be >= 1 MiB (got $CHUNK_SIZE)"
(( CHUNK_SIZE <= 20971520 ))    || log "⚠ --chunk-size $CHUNK_SIZE exceeds JSDelivr's 20 MB limit. OK for other CDNs (see README)."

# GGUF shard size
GGUF_SHARD_SIZE_BYTES="$(parse_size "$GGUF_SHARD_SIZE")"
[[ -z "$GGUF_SHARD_SIZE_BYTES" ]]        && die "--gguf-shard-size: cannot parse '$GGUF_SHARD_SIZE' (use e.g. 1800M, 1.5G)"
(( GGUF_SHARD_SIZE_BYTES >= 104857600 )) || die "--gguf-shard-size must be >= 100M (got $GGUF_SHARD_SIZE)"
(( GGUF_SHARD_SIZE_BYTES < WLLAMA_MAX )) || die "--gguf-shard-size must be < 2G for wllama (got $GGUF_SHARD_SIZE → $(human_size $GGUF_SHARD_SIZE_BYTES))"

# Conflicting flags
$MERGE && $OVERWRITE && die "cannot use both --merge and --overwrite"

# Output parent must exist
OUTPUT_PARENT="$(dirname "$(mkdir -p "$(dirname "$OUTPUT_DIR")" && cd "$(dirname "$OUTPUT_DIR")" && pwd)/$(basename "$OUTPUT_DIR")")"
[[ -d "$OUTPUT_PARENT" ]] || die "parent directory does not exist: $OUTPUT_PARENT"

# Writability check
if [[ -d "$OUTPUT_DIR" ]]; then
  [[ -w "$OUTPUT_DIR" ]] || die "output directory is not writable: $OUTPUT_DIR"
else
  [[ -w "$OUTPUT_PARENT" ]] || die "cannot create output directory — parent not writable: $OUTPUT_PARENT"
fi

# Inputs must exist
for input in "${INPUTS[@]}"; do
  [[ -e "$input" ]] || die "input not found: $input"
done

# gguf-split binary (if provided)
if [[ -n "$GGUF_SPLIT_BIN" ]]; then
  [[ -x "$GGUF_SPLIT_BIN" ]] || die "--gguf-split not found/executable: $GGUF_SPLIT_BIN"
fi

# =============================================================================
# Handle existing output directory
# =============================================================================
EXISTING_FILEMAP=""
_kv_setup
_KV_HASHES="$_KV_DIR/hashes.tsv"     # sha256 → cdn_file or "sharded"
_KV_CDN="$_KV_DIR/cdn.tsv"           # cdn_filename → virtual_path that owns it
_kv_init "$_KV_HASHES"
_kv_init "$_KV_CDN"
HAS_EXISTING=false            # true if output dir has a filemap.json

load_existing_filemap() {
  # Load existing filemap indexes for dedup + collision detection
  # Python writes tab-separated KV files directly (no eval, no declare -A)
  EXISTING_FILEMAP="$OUTPUT_DIR/filemap.json"
  python3 -c "
import json, sys
with open('$EXISTING_FILEMAP') as f:
    fm = json.load(f)
hf = open('$_KV_HASHES', 'w')
cf = open('$_KV_CDN', 'w')
for vp, entry in fm['files'].items():
    h = entry['sha256']
    if entry.get('shards'):
        hf.write(h + '\tsharded\n')
        for shard in entry['shards']:
            sf = shard['file']
            cf.write(sf + '\t' + vp + '\n')
    elif entry.get('cdn_file'):
        cdn = entry['cdn_file']
        hf.write(h + '\t' + cdn + '\n')
        cf.write(cdn + '\t' + vp + '\n')
hf.close(); cf.close()
"
}

if [[ -d "$OUTPUT_DIR" ]] && [[ -f "$OUTPUT_DIR/filemap.json" ]]; then
  HAS_EXISTING=true

  if $OVERWRITE; then
    log "Overwrite: wiping existing output ($OUTPUT_DIR)"
    rm -rf "$OUTPUT_DIR"
    HAS_EXISTING=false
  else
    # Load existing filemap — needed for both --merge AND the no-flag analysis
    load_existing_filemap
    log "Existing package found: $(_kv_count "$_KV_HASHES") hashes, $(_kv_count "$_KV_CDN") CDN files"
  fi

elif [[ -d "$OUTPUT_DIR" ]]; then
  existing_count=$(find "$OUTPUT_DIR" -maxdepth 1 -type f 2>/dev/null | wc -l)
  if (( existing_count > 0 )) && ! $OVERWRITE; then
    die "output directory has $existing_count files but no filemap.json.
  Use --overwrite to wipe, or clear it manually."
  fi
  if $OVERWRITE && (( existing_count > 0 )); then
    log "Overwrite: wiping existing output ($OUTPUT_DIR)"
    rm -rf "$OUTPUT_DIR"
  fi
fi

# =============================================================================
# Phase 0: Discover input files
# =============================================================================
FILE_VP=()
FILE_PP=()

for input in "${INPUTS[@]}"; do
  if [[ -d "$input" ]]; then
    input_dir="$(cd "$input" && pwd)"
    input_dir="${input_dir%/}/"
    log "Scanning folder: $input_dir"

    while IFS= read -r -d '' filepath; do
      rel="${filepath#$input_dir}"
      if should_exclude "$filepath"; then
        vlog "Excluded: $rel"
        continue
      fi
      FILE_VP+=("$rel")
      FILE_PP+=("$filepath")
      vlog "Found: $rel ($(human_size $(get_size "$filepath")))"
    done < <(find -L "$input_dir" -type f -print0 | sort -z)

  elif [[ -f "$input" ]]; then
    FILE_VP+=("$(basename "$input")")
    FILE_PP+=("$input")
  else
    die "input is neither a file nor directory: $input"
  fi
done

[[ ${#FILE_VP[@]} -eq 0 ]] && die "no files found after scanning inputs"

echo ""
echo "=== model-packager v${VERSION} ==="
echo "Output:     $OUTPUT_DIR (flat)"
echo "Chunk size: $(human_size $CHUNK_SIZE) ($CHUNK_SIZE bytes)"
$MERGE && echo "Mode:       MERGE (dedup via SHA256)"
echo "Discovered: ${#FILE_VP[@]} files"
echo ""

for i in "${!FILE_VP[@]}"; do
  size=$(get_size "${FILE_PP[$i]}")
  printf "  %-45s %10s\n" "${FILE_VP[$i]}" "$(human_size $size)"
done
echo ""

# =============================================================================
# Phase 1: GGUF split expansion
# =============================================================================
NEED_GGUF_SPLIT=false
for i in "${!FILE_PP[@]}"; do
  if is_gguf "${FILE_PP[$i]}" && needs_gguf_split "${FILE_PP[$i]}"; then
    NEED_GGUF_SPLIT=true; break
  fi
done

if $NEED_GGUF_SPLIT; then
  find_gguf_split || die "GGUF exceeding $(human_size $GGUF_SHARD_SIZE_BYTES) found but llama-gguf-split not in PATH. Use --gguf-split."
  log "Using gguf-split: $GGUF_SPLIT_BIN (max shard: $GGUF_SHARD_SIZE)"
fi

TMPDIR_GGUF=""
cleanup() {
  [[ -n "$TMPDIR_GGUF" ]] && [[ -d "$TMPDIR_GGUF" ]] && rm -rf "$TMPDIR_GGUF" || true
  [[ -n "$_KV_DIR" ]]     && [[ -d "$_KV_DIR" ]]     && rm -rf "$_KV_DIR"     || true
}
trap cleanup EXIT

DELIV_VP=()
DELIV_PP=()
DELIV_SRC=()

for i in "${!FILE_VP[@]}"; do
  vp="${FILE_VP[$i]}"
  pp="${FILE_PP[$i]}"

  if is_gguf "$pp" && needs_gguf_split "$pp"; then
    log "GGUF split: $vp ($(human_size $(get_size "$pp"))) → max shard $GGUF_SHARD_SIZE"

    if $DRY_RUN; then
      log "  [dry-run] Would run gguf-split --split-max-size $GGUF_SHARD_SIZE"
      # Estimate: still add as single deliverable for space calc
      DELIV_VP+=("$vp"); DELIV_PP+=("$pp"); DELIV_SRC+=("")
      continue
    fi

    [[ -z "$TMPDIR_GGUF" ]] && TMPDIR_GGUF="$(mktemp -d)"
    prefix="$(basename "${pp%.gguf}")"
    "$GGUF_SPLIT_BIN" --split-max-size "$GGUF_SHARD_SIZE" "$pp" "$TMPDIR_GGUF/$prefix" 2>&1 | \
      while IFS= read -r line; do vlog "  gguf-split: $line"; done

    for shard in "$TMPDIR_GGUF/"*-of-*.gguf; do
      [[ -f "$shard" ]] || continue
      shard_name="$(basename "$shard")"
      dir_prefix="$(dirname "$vp")"
      [[ "$dir_prefix" == "." ]] && shard_vp="$shard_name" || shard_vp="$dir_prefix/$shard_name"
      DELIV_VP+=("$shard_vp")
      DELIV_PP+=("$shard")
      DELIV_SRC+=("$vp")
      log "  → $shard_name ($(human_size $(get_size "$shard")))"
    done
  else
    DELIV_VP+=("$vp")
    DELIV_PP+=("$pp")
    # Detect pre-split GGUF shards (e.g. model-Q4_K_M-00001-of-00003.gguf)
    # and tag them with gguf_source for manifest grouping
    if is_gguf "$pp" && [[ "$(basename "$pp")" =~ -[0-9]{5}-of-[0-9]{5}\.gguf$ ]]; then
      base_name="$(basename "$pp" | sed 's/-[0-9]\{5\}-of-[0-9]\{5\}\.gguf$/.gguf/')"
      dir_prefix="$(dirname "$vp")"
      [[ "$dir_prefix" == "." ]] && logical_vp="$base_name" || logical_vp="$dir_prefix/$base_name"
      DELIV_SRC+=("$logical_vp")
      vlog "Pre-split GGUF: $vp → source: $logical_vp"
    else
      DELIV_SRC+=("")
    fi
  fi
done

# =============================================================================
# Phase 2: Free space check
# =============================================================================
total_input_bytes=0
for i in "${!DELIV_PP[@]}"; do
  s=$(get_size "${DELIV_PP[$i]}")
  total_input_bytes=$((total_input_bytes + s))
done

# In merge mode, files with matching SHA256 won't be written — but we can't
# cheaply compute all hashes upfront for large files. Use total as upper bound
# and add ~1% overhead for filemap.json + shard metadata.
space_needed=$((total_input_bytes + total_input_bytes / 100))

# For gguf-split, the split output is roughly equal to input (negligible header
# overhead, <1KB per shard). Temp dir needs space for the largest single GGUF.
largest_gguf_bytes=0
for i in "${!FILE_PP[@]}"; do
  if is_gguf "${FILE_PP[$i]}" && needs_gguf_split "${FILE_PP[$i]}"; then
    s=$(get_size "${FILE_PP[$i]}")
    (( s > largest_gguf_bytes )) && largest_gguf_bytes=$s
  fi
done
# Temp dir (if gguf-split used) is on same volume as /tmp
if (( largest_gguf_bytes > 0 )); then
  tmpdir_vol="${TMPDIR:-/tmp}"
  tmp_free=$(get_free_space "$tmpdir_vol")
  if (( tmp_free < largest_gguf_bytes )); then
    die "insufficient temp space for gguf-split.
  Need: $(human_size $largest_gguf_bytes) for largest GGUF
  Free: $(human_size $tmp_free) on $(df -P "$tmpdir_vol" | awk 'NR==2{print $6}')
  Set TMPDIR to a volume with more space."
  fi
fi

# Check output volume
output_check_path="$OUTPUT_DIR"
[[ -d "$output_check_path" ]] || output_check_path="$OUTPUT_PARENT"
output_free=$(get_free_space "$output_check_path")

if (( output_free < space_needed )); then
  die "insufficient disk space on output volume.
  Need: ~$(human_size $space_needed) (upper bound)
  Free: $(human_size $output_free) on $(df -P "$output_check_path" | awk 'NR==2{print $6}')"
fi
vlog "Space check OK: need ~$(human_size $space_needed), have $(human_size $output_free)"

$DRY_RUN && { echo ""; echo "[dry-run] ${#DELIV_VP[@]} deliverables, ~$(human_size $space_needed) needed"; exit 0; }

# =============================================================================
# Phase 3: Compute flat CDN names + SHA256 dedup + collision detection
# =============================================================================
mkdir -p "$OUTPUT_DIR"

_KV_FLAT="$_KV_DIR/flat.tsv"  # flat_name → first claimant index (within this run)
_kv_init "$_KV_FLAT"
DELIV_FLAT=()             # flat CDN name per deliverable
DELIV_HASH=()             # SHA256 per deliverable
DELIV_SKIP=()             # "yes"|"sharded" if deduped, "no" otherwise
DELIV_DEDUP_CDN=()        # existing cdn_file to reference (for dedup)

dedup_count=0
dedup_bytes=0
new_file_count=0
new_file_bytes=0
merge_conflicts=()

for i in "${!DELIV_VP[@]}"; do
  pp="${DELIV_PP[$i]}"
  vp="${DELIV_VP[$i]}"
  hash=$(file_sha256 "$pp")
  DELIV_HASH+=("$hash")

  # --- Check SHA256 dedup against existing filemap ---
  if $HAS_EXISTING && _kv_has "$_KV_HASHES" "$hash"; then
    existing_cdn="$(_kv_get "$_KV_HASHES" "$hash")"
    if [[ "$existing_cdn" == "sharded" ]]; then
      DELIV_SKIP+=("sharded")
      DELIV_DEDUP_CDN+=("")
    else
      DELIV_SKIP+=("yes")
      DELIV_DEDUP_CDN+=("$existing_cdn")
    fi
    s=$(get_size "$pp")
    dedup_count=$((dedup_count + 1))
    dedup_bytes=$((dedup_bytes + s))
    vlog "Dedup: $vp (SHA256 match → $existing_cdn)"
    DELIV_FLAT+=("")  # placeholder
    continue
  fi

  DELIV_SKIP+=("no")
  DELIV_DEDUP_CDN+=("")

  s=$(get_size "$pp")
  new_file_count=$((new_file_count + 1))
  new_file_bytes=$((new_file_bytes + s))

  # --- Compute flat CDN name ---
  bn="$(basename "$pp")"

  # Check if this flat name (or its shard prefix) collides with existing CDN files
  if $HAS_EXISTING; then
    # Direct file collision
    if _kv_has "$_KV_CDN" "$bn"; then
      existing_owner="$(_kv_get "$_KV_CDN" "$bn")"
      merge_conflicts+=("  '$bn' → existing: '$existing_owner', new: '$vp' (SHA256: ${hash:0:16}...)")
    fi
    # Shard prefix collision
    if needs_byte_split "$pp"; then
      shard_prefix="${bn}.shard."
      while IFS=$'\t' read -r existing_cdn_file existing_owner_val; do
        if [[ "$existing_cdn_file" == "${shard_prefix}"* ]]; then
          merge_conflicts+=("  '${bn}.shard.*' → existing: '$existing_owner_val', new: '$vp' (SHA256: ${hash:0:16}...)")
          break
        fi
      done < "$_KV_CDN"
    fi
  fi

  # Within-run collision check
  if ! _kv_has "$_KV_FLAT" "$bn"; then
    _kv_set "$_KV_FLAT" "$bn" "$i"
    DELIV_FLAT+=("$bn")
  else
    prev_idx="$(_kv_get "$_KV_FLAT" "$bn")"
    prev_vp="${DELIV_VP[$prev_idx]}"
    merge_conflicts+=("  '$bn' → two inputs in this run: '$prev_vp' and '$vp'")
    DELIV_FLAT+=("$bn")  # placeholder
  fi
done

# =============================================================================
# Phase 3b: Decision point — act on analysis results
# =============================================================================
if $HAS_EXISTING && ! $MERGE; then
  # No flag specified, but output has existing data — report analysis and exit
  echo ""
  echo "=== Existing package analysis ==="

  existing_vp_count=$(python3 -c "
import json
with open('$EXISTING_FILEMAP') as f:
    print(len(json.load(f)['files']))
")
  existing_file_count=$(find "$OUTPUT_DIR" -maxdepth 1 -type f | wc -l)
  echo "  Existing:  $existing_vp_count virtual paths, $existing_file_count CDN files"
  echo "  New input: ${#DELIV_VP[@]} files"

  if (( dedup_count > 0 )); then
    echo "  Dedup:     $dedup_count files match existing ($(human_size $dedup_bytes) — would be skipped)"
  fi
  echo "  New:       $new_file_count files with new content ($(human_size $new_file_bytes))"

  if [[ ${#merge_conflicts[@]} -gt 0 ]]; then
    echo ""
    echo "  ✗ MERGE NOT POSSIBLE — ${#merge_conflicts[@]} CDN filename collision(s):"
    echo ""
    for line in "${merge_conflicts[@]}"; do
      echo "  $line"
    done
    echo ""
    echo "  These files have different content but the same CDN filename."
    echo "  The Service Worker cannot serve two different files from the same name."
    echo ""
    echo "  Options:"
    echo "    --overwrite     Wipe existing package and start fresh"
    echo "    Separate -o     Use a different output directory (separate CDN repo)"
    echo "    Rename inputs   Rename the conflicting files before packaging"
  else
    echo ""
    echo "  ✓ MERGE POSSIBLE — no CDN filename collisions detected."
    echo "    $dedup_count file(s) would be deduplicated (same SHA256, stored once)."
    echo "    $new_file_count file(s) would be added as new."
    echo ""
    echo "  Options:"
    echo "    --merge         Add new files, dedup shared files, single filemap.json"
    echo "    --overwrite     Wipe existing package and start fresh"
  fi
  echo ""
  exit 0
fi

# If merge mode: die on collisions
if $MERGE && [[ ${#merge_conflicts[@]} -gt 0 ]]; then
  echo "" >&2
  echo "Error: CDN filename collision — merge is not safe." >&2
  echo "" >&2
  for line in "${merge_conflicts[@]}"; do
    echo "$line" >&2
  done
  echo "" >&2
  echo "Options:" >&2
  echo "  --overwrite           Wipe existing package and start fresh" >&2
  echo "  Separate -o           Use a different output directory" >&2
  echo "  Rename inputs         Rename conflicting files before packaging" >&2
  exit 1
fi

if (( dedup_count > 0 )); then
  log "Dedup: $dedup_count files skipped ($(human_size $dedup_bytes) saved, same SHA256)"
fi

# =============================================================================
# Phase 4: Write files + build filemap entries
# =============================================================================
log "Writing to $OUTPUT_DIR ..."

FILEMAP_ENTRIES=""

for i in "${!DELIV_VP[@]}"; do
  vp="${DELIV_VP[$i]}"
  pp="${DELIV_PP[$i]}"
  src="${DELIV_SRC[$i]}"
  hash="${DELIV_HASH[$i]}"
  skip="${DELIV_SKIP[$i]}"
  file_size=$(get_size "$pp")

  src_field=""
  [[ -n "$src" ]] && src_field=",\"gguf_source\":\"${src}\""

  if [[ "$skip" == "yes" ]]; then
    # Deduped unsharded file — reference existing cdn_file
    cdn="${DELIV_DEDUP_CDN[$i]}"
    [[ -n "$FILEMAP_ENTRIES" ]] && FILEMAP_ENTRIES+=","
    FILEMAP_ENTRIES+="\"${vp}\":{\"size\":${file_size},\"sha256\":\"${hash}\",\"cdn_file\":\"${cdn}\"${src_field},\"shards\":null}"
    vlog "Dedup ref: $vp → $cdn"
    continue
  fi

  if [[ "$skip" == "sharded" ]]; then
    # Deduped sharded file — find existing entry with same hash and copy its shard list
    shard_json=$(python3 -c "
import json
with open('$EXISTING_FILEMAP') as f:
    fm = json.load(f)
for vp_e, entry in fm['files'].items():
    if entry['sha256'] == '$hash' and entry.get('shards'):
        print(json.dumps(entry['shards']))
        break
")
    [[ -n "$FILEMAP_ENTRIES" ]] && FILEMAP_ENTRIES+=","
    FILEMAP_ENTRIES+="\"${vp}\":{\"size\":${file_size},\"sha256\":\"${hash}\",\"cdn_file\":null${src_field},\"shards\":${shard_json}}"
    vlog "Dedup ref (sharded): $vp"
    continue
  fi

  fn="${DELIV_FLAT[$i]}"

  if needs_byte_split "$pp"; then
    shard_count=$(( (file_size + CHUNK_SIZE - 1) / CHUNK_SIZE ))
    log "Shard:  $vp → ${fn}.shard.* ($shard_count shards)"

    split -b "$CHUNK_SIZE" -d -a 3 "$pp" "$OUTPUT_DIR/${fn}.shard."

    if $KEEP_INTERMEDIATES && [[ -n "$src" ]]; then
      cp "$pp" "$OUTPUT_DIR/$fn"
    fi

    shard_json=""
    shard_offset=0
    for sf in "$OUTPUT_DIR/${fn}".shard.*; do
      sfn="$(basename "$sf")"
      ss=$(get_size "$sf")
      sh=$(file_sha256 "$sf")
      [[ -n "$shard_json" ]] && shard_json+=","
      shard_json+="{\"file\":\"${sfn}\",\"offset\":${shard_offset},\"size\":${ss},\"sha256\":\"${sh}\"}"
      shard_offset=$((shard_offset + ss))
    done

    [[ -n "$FILEMAP_ENTRIES" ]] && FILEMAP_ENTRIES+=","
    FILEMAP_ENTRIES+="\"${vp}\":{\"size\":${file_size},\"sha256\":\"${hash}\",\"cdn_file\":null${src_field},\"shards\":[${shard_json}]}"

    # Register hash for dedup within this run
    _kv_set "$_KV_HASHES" "$hash" "sharded"
  else
    log "Direct: $vp → $fn"
    cp "$pp" "$OUTPUT_DIR/$fn"

    [[ -n "$FILEMAP_ENTRIES" ]] && FILEMAP_ENTRIES+=","
    FILEMAP_ENTRIES+="\"${vp}\":{\"size\":${file_size},\"sha256\":\"${hash}\",\"cdn_file\":\"${fn}\"${src_field},\"shards\":null}"

    # Register hash for dedup within this run
    _kv_set "$_KV_HASHES" "$hash" "$fn"
  fi
done

# =============================================================================
# Phase 5: Write filemap.json (merge with existing if needed)
# =============================================================================
if $MERGE && [[ -n "$EXISTING_FILEMAP" ]]; then
  python3 -c "
import json

# Load existing
with open('$EXISTING_FILEMAP') as f:
    existing = json.load(f)

# Parse new entries
new_raw = '{\"files\":{${FILEMAP_ENTRIES}}}'
new_data = json.loads(new_raw)

# Merge: new entries override existing on virtual path collision
existing['files'].update(new_data['files'])
existing['generator'] = 'model-packager v${VERSION}'
existing['created'] = '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
existing['version'] = 5

with open('$OUTPUT_DIR/filemap.json', 'w') as f:
    json.dump(existing, f, indent=2)
    f.write('\n')

total = len(existing['files'])
added = len(new_data['files'])
print(f'  Merged filemap.json: {added} new + {total - added} existing = {total} total virtual paths')
"
else
  python3 -c "
import json
raw = '{\"files\":{${FILEMAP_ENTRIES}}}'
data = json.loads(raw)
output = {
    'version': 5,
    'generator': 'model-packager v${VERSION}',
    'created': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
    'chunk_size': ${CHUNK_SIZE},
    'files': data['files']
}
with open('$OUTPUT_DIR/filemap.json', 'w') as f:
    json.dump(output, f, indent=2)
    f.write('\n')
print(f'  Wrote filemap.json ({len(data[\"files\"])} virtual paths)')
"
fi


# =============================================================================
# Phase 5b: Generate manifests (with GGUF metadata extraction)
# =============================================================================
# Two modes:
#   --manifest NAME: all files from THIS run → named manifest.
#   (no --manifest): auto-detect groups from file extensions/patterns.
#
# For GGUF files, if gguf-meta.py is present:
#   - Extracts metadata (architecture, quantization, context length, etc.)
#   - Classifies each GGUF as "llm" or "mmproj"
#   - Auto-generates cross-permutation manifests (LLM-only + LLM+mmproj combos)
#   - Stores metadata in filemap.json under "gguf_metadata"
#
# In merge mode, existing manifests are preserved; new ones are added/updated.
GGUF_META_PY="$SCRIPT_DIR/gguf-meta.py"
python3 -c "
import json, re, sys, os, subprocess

manifest_name = '$MANIFEST_NAME'
is_merge = '$MERGE' == 'true'
output_dir = '$OUTPUT_DIR'
gguf_meta_py = '$GGUF_META_PY'

with open(os.path.join(output_dir, 'filemap.json')) as f:
    fm = json.load(f)

files = fm.get('files', {})
if not files:
    sys.exit(0)

existing_manifests = fm.get('manifests', {}) if is_merge else {}

# ── Determine which virtual paths were added by THIS packaging run ──
all_vps = set(files.keys())
existing_vps = set()
for m in existing_manifests.values():
    existing_vps.update(m.get('files', []))
new_vps = all_vps - existing_vps if is_merge else all_vps

# ── GGUF metadata extraction helper ──
has_gguf_meta = os.path.isfile(gguf_meta_py)

def extract_gguf_metadata(vp, entry):
    '''Read GGUF metadata from the first CDN shard (header is at file start).'''
    if not has_gguf_meta:
        return None
    # Find the actual file to read
    if 'shards' in entry and entry['shards']:
        fpath = os.path.join(output_dir, entry['shards'][0]['file'])
    elif 'cdn_file' in entry:
        fpath = os.path.join(output_dir, entry['cdn_file'])
    else:
        fpath = os.path.join(output_dir, vp)
    if not os.path.isfile(fpath):
        return None
    try:
        result = subprocess.run(
            [sys.executable, gguf_meta_py, '--json', fpath],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            # Get the first (and only) entry
            for name, info in data.items():
                return info
    except Exception:
        pass
    return None

if manifest_name:
    # ── Explicit manifest mode ──
    manifest_files = sorted(all_vps) if not is_merge else sorted(new_vps | {
        vp for vp in all_vps
        if not any(vp.endswith(ext) for ext in ('.onnx', '.onnx_data', '.gguf'))
    })
    manifest_size = sum(files[vp]['size'] for vp in manifest_files if vp in files)
    manifests = dict(existing_manifests)
    manifests[manifest_name] = { 'files': manifest_files, 'size': manifest_size }
else:
    # ── Auto-detect mode ──
    shared = []
    onnx_groups = {}

    # GGUF: classify each file individually, THEN group into LLM vs mmproj
    gguf_llm = {}     # quant → { 'files': [...], 'meta': {...} }
    gguf_mmproj = {}  # quant → { 'files': [...], 'meta': {...} }
    gguf_metadata_all = {}  # stored in filemap.json

    # First pass: group shards by gguf_source (pre-split GGUFs share a source)
    gguf_source_groups = {}  # source_key → [vp, ...]
    gguf_non_shard = []       # standalone GGUF VPs

    for vp, entry in files.items():
        if vp.endswith('.onnx') or vp.endswith('.onnx_data'):
            basename = vp.rsplit('/', 1)[-1]
            stem = re.sub(r'\.onnx(_data)?\$', '', basename)
            name = re.sub(r'^model_?', '', stem) or 'default'
            onnx_groups.setdefault(name, []).append(vp)
        elif vp.endswith('.gguf'):
            src = entry.get('gguf_source', '')
            if src:
                gguf_source_groups.setdefault(src, []).append(vp)
            else:
                gguf_non_shard.append(vp)
        else:
            shared.append(vp)

    # Second pass: classify each logical GGUF and route to LLM or mmproj
    def process_gguf_group(vp_list, source_label):
        '''Classify a group of GGUF VPs (single file or pre-split shards) and route.'''
        first_vp = vp_list[0]
        meta = extract_gguf_metadata(first_vp, files[first_vp])

        # Determine quant from filename or metadata
        src_base = source_label.rsplit('/', 1)[-1].replace('.gguf', '')
        src_base = re.sub(r'-\d{5}-of-\d{5}\$', '', src_base)
        m_q = re.search(r'[-_]((?:Q|F|IQ|BF)\d[A-Za-z0-9_]*)', src_base, re.IGNORECASE)
        quant_from_name = m_q.group(1) if m_q else src_base

        classification = 'llm'
        quant = quant_from_name

        if meta and not meta.get('error'):
            classification = meta.get('classification', 'llm')
            meta_quant = meta.get('quantization')
            if meta_quant and meta_quant != 'unknown':
                quant = meta_quant

            meta_key = source_label.rsplit('/', 1)[-1]
            stored_meta = {k: v for k, v in meta.items()
                          if k not in ('arch_params', 'file', 'file_size')}
            gguf_metadata_all[meta_key] = stored_meta

        target = gguf_mmproj if classification == 'mmproj' else gguf_llm
        norm_name = quant.lower()
        if norm_name not in target:
            target[norm_name] = {'files': [], 'meta': meta, 'display_quant': quant}
        target[norm_name]['files'].extend(vp_list)

    # Process pre-split shard groups
    for src, vps in gguf_source_groups.items():
        process_gguf_group(vps, src)

    # Process standalone GGUFs
    for vp in gguf_non_shard:
        process_gguf_group([vp], vp)

    # ── Build manifests ──
    manifests = dict(existing_manifests)

    # ONNX manifests (unchanged logic)
    for name, group_files in sorted(onnx_groups.items()):
        manifest_files = sorted(set(shared + group_files))
        manifest_size = sum(files[vp]['size'] for vp in manifest_files if vp in files)
        manifests[name] = {'files': manifest_files, 'size': manifest_size}

    # GGUF LLM-only manifests
    for norm_name, group in sorted(gguf_llm.items()):
        manifest_files = sorted(set(shared + group['files']))
        manifest_size = sum(files[vp]['size'] for vp in manifest_files if vp in files)
        manifests[norm_name] = {'files': manifest_files, 'size': manifest_size}

    # GGUF mmproj-only manifests (for completeness — rarely loaded alone)
    for norm_name, group in sorted(gguf_mmproj.items()):
        mname = f'mmproj_{norm_name}'
        manifest_files = sorted(set(shared + group['files']))
        manifest_size = sum(files[vp]['size'] for vp in manifest_files if vp in files)
        manifests[mname] = {'files': manifest_files, 'size': manifest_size}

    # GGUF cross-permutation manifests: each LLM × each mmproj
    cross_combos = []
    if gguf_llm and gguf_mmproj:
        for llm_name, llm_group in sorted(gguf_llm.items()):
            for mm_name, mm_group in sorted(gguf_mmproj.items()):
                combo_name = f'{llm_name}+mmproj_{mm_name}'
                combo_files = sorted(set(shared + llm_group['files'] + mm_group['files']))
                combo_size = sum(files[vp]['size'] for vp in combo_files if vp in files)
                manifests[combo_name] = {'files': combo_files, 'size': combo_size}
                cross_combos.append(combo_name)

    # If no groups at all, nothing to do
    if not manifests and not existing_manifests:
        sys.exit(0)

# ── Store GGUF metadata in filemap ──
if gguf_metadata_all:
    fm['gguf_metadata'] = gguf_metadata_all

fm['manifests'] = manifests

with open(os.path.join(output_dir, 'filemap.json'), 'w') as f:
    json.dump(fm, f, indent=2)
    f.write('\n')

# ── Summary output ──
new_names = [n for n in manifests if n not in existing_manifests]
kept = len(existing_manifests)
desc = ', '.join(f'{n} ({len(manifests[n][\"files\"])} files, {manifests[n][\"size\"]/1048576:.1f} MB)' for n in sorted(manifests))
summary = f'  Manifests: {desc}'
if kept: summary += f' ({kept} preserved from previous)'
print(summary)

# ── GGUF metadata summary ──
if gguf_metadata_all:
    for key, meta in gguf_metadata_all.items():
        cls = meta.get('classification', '?')
        arch = meta.get('architecture', '?')
        quant = meta.get('quantization', '?')
        ctx = meta.get('context_length', '?')
        blocks = meta.get('block_count', '?')
        embd = meta.get('embedding_length', '?')
        vocab = meta.get('vocab_size', '?')
        print(f'  GGUF [{cls}] {key}: arch={arch} quant={quant} ctx={ctx} layers={blocks} embd={embd} vocab={vocab}')

# ── Cross-permutation warnings ──
if cross_combos:
    n_llm = len(gguf_llm)
    n_mm = len(gguf_mmproj)
    n_combos = len(cross_combos)
    print(f'  ⚠ Generated {n_combos} LLM+mmproj cross-permutation manifest(s) from {n_llm} LLM × {n_mm} mmproj.')
    print(f'    Some permutations may be invalid (architecture mismatch, incompatible quant).')
    print(f'    Review manifests in filemap.json and remove any invalid combinations.')
    for combo in cross_combos:
        print(f'      {combo}')
"

# Phase 6: Summary + verification
# =============================================================================
echo ""
echo "=== Package Summary ==="
total_files=$(find "$OUTPUT_DIR" -maxdepth 1 -type f | wc -l)
total_size=$(du -sb "$OUTPUT_DIR" | cut -f1)
echo "  Output:     $OUTPUT_DIR"
echo "  Files:      $total_files (flat)"
echo "  Total size: $(human_size $total_size)"
if (( dedup_count > 0 )); then
  echo "  Deduped:    $dedup_count files ($(human_size $dedup_bytes) saved)"
fi

oversized=0
while IFS= read -r -d '' f; do
  bn="$(basename "$f")"
  [[ "$bn" == "filemap.json" ]] && continue
  s=$(get_size "$f")
  if (( s > CHUNK_SIZE )); then
    if $KEEP_INTERMEDIATES && [[ "$bn" == *-of-*.gguf ]]; then :
    else
      echo "  ⚠ Over CDN limit: $bn ($(human_size $s))"
      oversized=$((oversized + 1))
    fi
  fi
done < <(find "$OUTPUT_DIR" -maxdepth 1 -type f -print0)
if (( oversized == 0 )); then echo "  ✓ All CDN files under $(human_size $CHUNK_SIZE)"; fi

if $REMOVE_ORIGINALS; then
  echo ""; log "Removing originals..."
  for input in "${INPUTS[@]}"; do
    if [[ -d "$input" ]]; then rm -rv "$input"
    elif [[ -f "$input" ]]; then rm -v "$input"
    fi
  done
fi

echo ""
echo "Done."
