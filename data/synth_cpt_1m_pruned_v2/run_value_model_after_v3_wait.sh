#!/usr/bin/env bash
set -euo pipefail

WORK=/root/rivermind-data/qwen_ag_lm
. "$WORK/env.sh" 2>/dev/null || true
. "$WORK/venv/bin/activate" 2>/dev/null || true
cd "$WORK"

WAIT_TAG=${WAIT_TAG:-unsolved_high_budget_value_v3_cost_template_depth16_v1}
VALUE_TAG=${VALUE_TAG:-v5_timeout_hardneg_features_v1_plus_v3}
WAIT_INTERVAL=${WAIT_INTERVAL:-300}
OUT_DIR=${OUT_DIR:-outputs/candidate_value_model_${VALUE_TAG}}
LOG=${LOG:-outputs/${VALUE_TAG}.wait_then_train_value.log}
DRY_RUN=${DRY_RUN:-0}

V1_EVENTS_DIR=${V1_EVENTS_DIR:-outputs/final_eval_imo_ag30_qwen_v1/events}
V1_SUMMARY_JSONL=${V1_SUMMARY_JSONL:-outputs/final_eval_imo_ag30_qwen_v1/summary.jsonl}
V3_EVENTS_DIR=${V3_EVENTS_DIR:-outputs/final_eval_imo_ag30_qwen_unsolved_high_budget_value_v3_cost_template_depth16_v1/events}
V3_SUMMARY_JSONL=${V3_SUMMARY_JSONL:-outputs/final_eval_imo_ag30_qwen_unsolved_high_budget_value_v3_cost_template_depth16_v1/summary.jsonl}

V1_DATA=$OUT_DIR/candidate_value_data_v1.jsonl
V3_DATA=$OUT_DIR/candidate_value_data_v3.jsonl
MERGED_DATA=$OUT_DIR/candidate_value_data.jsonl
MODEL_FILE=$OUT_DIR/candidate_value_model.json
SUMMARY_FILE=$OUT_DIR/summary.json

mkdir -p "$OUT_DIR"
export WAIT_TAG VALUE_TAG OUT_DIR V1_EVENTS_DIR V1_SUMMARY_JSONL V3_EVENTS_DIR V3_SUMMARY_JSONL

if [ "$DRY_RUN" = "1" ]; then
  python - <<'PY'
import json
import os
print(json.dumps({
    'wait_tag': os.environ.get('WAIT_TAG', 'unsolved_high_budget_value_v3_cost_template_depth16_v1'),
    'value_tag': os.environ.get('VALUE_TAG', 'v5_timeout_hardneg_features_v1_plus_v3'),
    'out_dir': os.environ.get('OUT_DIR'),
    'v1_events_dir': os.environ.get('V1_EVENTS_DIR'),
    'v1_summary_jsonl': os.environ.get('V1_SUMMARY_JSONL'),
    'v3_events_dir': os.environ.get('V3_EVENTS_DIR'),
    'v3_summary_jsonl': os.environ.get('V3_SUMMARY_JSONL'),
}, ensure_ascii=False, indent=2))
PY
  exit 0
fi

echo "waiting for run tag: $WAIT_TAG" | tee -a "$LOG"
while pgrep -f "$WAIT_TAG" >/dev/null; do
  date '+%F %T %z' | tee -a "$LOG"
  pgrep -af "$WAIT_TAG" | tee -a "$LOG" || true
  sleep "$WAIT_INTERVAL"
done

echo "building v1 value rows" | tee -a "$LOG"
python -u scripts/build_candidate_value_data.py \
  --script_dir scripts \
  --events_dir "$V1_EVENTS_DIR" \
  --summary_jsonl "$V1_SUMMARY_JSONL" \
  --out_file "$V1_DATA" \
  > "$OUT_DIR/build_v1.log" 2>&1

echo "building v3 value rows" | tee -a "$LOG"
python -u scripts/build_candidate_value_data.py \
  --script_dir scripts \
  --events_dir "$V3_EVENTS_DIR" \
  --summary_jsonl "$V3_SUMMARY_JSONL" \
  --out_file "$V3_DATA" \
  > "$OUT_DIR/build_v3.log" 2>&1

python - "$V1_DATA" "$V3_DATA" "$MERGED_DATA" "$SUMMARY_FILE" <<'PY'
import json
import sys
from pathlib import Path

v1, v3, merged, summary = map(Path, sys.argv[1:])
rows = []
source_counts = {}
candidate_source_counts = {}
reason_counts = {}
hard_negative_counts = {}
for source, path in [('v1', v1), ('v3', v3)]:
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row['value_data_source'] = source
        rows.append(row)
        source_counts[source] = source_counts.get(source, 0) + 1
        candidate_source = row.get('source') or 'lm'
        candidate_source_counts[candidate_source] = candidate_source_counts.get(candidate_source, 0) + 1
        reason = row.get('reason')
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        error_text = str(row.get('candidate_ddar_error') or '').lower()
        if 'timeout' in error_text:
            hard_negative_counts['timeout'] = hard_negative_counts.get('timeout', 0) + 1
        if reason in {'point_too_close', 'point_too_far', 'point_already_exists', 'timeout', 'candidate_ddar_error'}:
            hard_negative_counts[reason] = hard_negative_counts.get(reason, 0) + 1
merged.write_text(
    ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
    encoding='utf-8',
)
summary_obj = {
    'rows': len(rows),
    'source_counts': source_counts,
    'candidate_source_counts': candidate_source_counts,
    'reason_counts': reason_counts,
    'hard_negative_counts': hard_negative_counts,
    'positives': sum(int(row.get('label', 0)) for row in rows),
    'negatives': len(rows) - sum(int(row.get('label', 0)) for row in rows),
}
summary.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary_obj, ensure_ascii=False, indent=2))
if not rows:
    raise SystemExit('no value rows built')
PY

echo "training value model: $VALUE_TAG" | tee -a "$LOG"
python -u scripts/train_candidate_value_model.py \
  --train_file "$MERGED_DATA" \
  --out_file "$MODEL_FILE" \
  > "$OUT_DIR/train.log" 2>&1

cat "$SUMMARY_FILE"
python - "$MODEL_FILE" <<'PY'
import json
import sys
from pathlib import Path
model = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(json.dumps({
    'model_file': sys.argv[1],
    'label_counts': model.get('label_counts'),
    'metrics': model.get('metrics'),
    'warnings': model.get('warnings'),
}, ensure_ascii=False, indent=2))
PY
