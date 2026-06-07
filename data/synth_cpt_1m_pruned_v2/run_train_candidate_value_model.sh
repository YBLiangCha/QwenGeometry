#!/usr/bin/env bash
set -euo pipefail

WORK=/root/rivermind-data/qwen_ag_lm
. "$WORK/env.sh" 2>/dev/null || true
. "$WORK/venv/bin/activate" 2>/dev/null || true
cd "$WORK"

VALUE_TAG=${VALUE_TAG:-v1_from_qwen_logs}
EVENTS_DIR=${EVENTS_DIR:-outputs/final_eval_imo_ag30_qwen_v1/events}
SUMMARY_JSONL=${SUMMARY_JSONL:-outputs/final_eval_imo_ag30_qwen_v1/summary.jsonl}
OUT_DIR=outputs/candidate_value_model_${VALUE_TAG}
DATA_FILE=$OUT_DIR/candidate_value_data.jsonl
MODEL_FILE=$OUT_DIR/candidate_value_model.json
mkdir -p "$OUT_DIR"

python -u scripts/build_candidate_value_data.py \
  --script_dir scripts \
  --events_dir "$EVENTS_DIR" \
  --summary_jsonl "$SUMMARY_JSONL" \
  --out_file "$DATA_FILE" \
  > "$OUT_DIR/build_data.log" 2>&1

python -u scripts/train_candidate_value_model.py \
  --train_file "$DATA_FILE" \
  --out_file "$MODEL_FILE" \
  > "$OUT_DIR/train.log" 2>&1

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
