#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:-/root/rivermind-data/qwen_ag_lm}
. "$WORK/env.sh" 2>/dev/null || true
. "$WORK/venv/bin/activate" 2>/dev/null || true
cd "$WORK"

VALUE_TAG=${VALUE_TAG:-v6_v5_plus_semantic_v3_partial4_typed_v1}
SCRIPT_DIR=${SCRIPT_DIR:-scripts}
BASE_VALUE_DATA=${BASE_VALUE_DATA:-outputs/candidate_value_model_v5_timeout_hardneg_features_v1_plus_v3/candidate_value_data.jsonl}
PARTIAL_TAG=${PARTIAL_TAG:-unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1}
PARTIAL_OUT_DIR=${PARTIAL_OUT_DIR:-outputs/final_eval_imo_ag30_qwen_${PARTIAL_TAG}}
PARTIAL_EVENTS_DIR=${PARTIAL_EVENTS_DIR:-${PARTIAL_OUT_DIR}/events}
PARTIAL_SUMMARY_JSONL=${PARTIAL_SUMMARY_JSONL:-${PARTIAL_OUT_DIR}/summary.jsonl}
OUT_DIR=${OUT_DIR:-outputs/candidate_value_model_${VALUE_TAG}}
PARTIAL_DATA=${PARTIAL_DATA:-${OUT_DIR}/candidate_value_data_partial.jsonl}
MERGED_DATA=${MERGED_DATA:-${OUT_DIR}/candidate_value_data.jsonl}
MODEL_FILE=${MODEL_FILE:-${OUT_DIR}/candidate_value_model.json}
SUMMARY_FILE=${SUMMARY_FILE:-${OUT_DIR}/summary.json}
INCLUDE_UNEVALUATED_VALID=${INCLUDE_UNEVALUATED_VALID:-1}

mkdir -p "$OUT_DIR"

if [ ! -s "$BASE_VALUE_DATA" ]; then
  echo "missing BASE_VALUE_DATA: $BASE_VALUE_DATA" >&2
  exit 1
fi

BUILD_ARGS=(
  --script_dir "$SCRIPT_DIR"
  --events_dir "$PARTIAL_EVENTS_DIR"
  --summary_jsonl "$PARTIAL_SUMMARY_JSONL"
  --out_file "$PARTIAL_DATA"
)
if [ "$INCLUDE_UNEVALUATED_VALID" = "1" ]; then
  BUILD_ARGS+=(--include_unevaluated_valid)
fi

echo "building partial value rows: $PARTIAL_TAG"
python -u "$SCRIPT_DIR/build_candidate_value_data.py" "${BUILD_ARGS[@]}" \
  > "$OUT_DIR/build_partial.log" 2>&1

echo "merging base and partial value rows"
python - "$BASE_VALUE_DATA" "$PARTIAL_DATA" "$MERGED_DATA" "$SUMMARY_FILE" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

base_path, partial_path, merged_path, summary_path = map(Path, sys.argv[1:])
rows = []
seen = set()
source_counts = Counter()
candidate_source_counts = Counter()
reason_counts = Counter()
label_counts = Counter()
construction_counts = Counter()

def row_key(row):
  return (
      row.get('value_data_source'),
      row.get('problem'),
      row.get('depth'),
      row.get('raw'),
      row.get('translation'),
      row.get('reason'),
  )

def add_rows(path, value_source):
  for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
    if not line.strip():
      continue
    row = json.loads(line)
    row.setdefault('value_data_source', value_source)
    key = row_key(row)
    if key in seen:
      continue
    seen.add(key)
    rows.append(row)
    source_counts[row.get('value_data_source') or 'unknown'] += 1
    candidate_source_counts[row.get('source') or 'lm'] += 1
    reason_counts[row.get('reason') or 'unknown'] += 1
    label_counts[str(int(row.get('label', 0)))] += 1
    construction_counts[row.get('construction_type') or 'unknown'] += 1

add_rows(base_path, 'base')
add_rows(partial_path, 'semantic_v3_partial')
merged_path.write_text(
    ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
    encoding='utf-8',
)
summary = {
    'base_value_data': str(base_path),
    'partial_value_data': str(partial_path),
    'merged_data': str(merged_path),
    'rows': len(rows),
    'source_counts': dict(source_counts),
    'candidate_source_counts': dict(candidate_source_counts),
    'reason_counts': dict(reason_counts),
    'label_counts': dict(label_counts),
    'construction_types_top': dict(construction_counts.most_common(20)),
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
if not rows:
  raise SystemExit('no merged value rows')
PY

echo "training value model: $VALUE_TAG"
python -u "$SCRIPT_DIR/train_candidate_value_model.py" \
  --train_file "$MERGED_DATA" \
  --out_file "$MODEL_FILE" \
  > "$OUT_DIR/train.log" 2>&1

cat "$SUMMARY_FILE"
python - "$MODEL_FILE" <<'PY'
import json
import sys
from pathlib import Path

model_file = Path(sys.argv[1])
model = json.loads(model_file.read_text(encoding='utf-8'))
print(json.dumps({
    'model_file': str(model_file),
    'label_counts': model.get('label_counts'),
    'metrics': model.get('metrics'),
    'warnings': model.get('warnings'),
}, ensure_ascii=False, indent=2))
PY
