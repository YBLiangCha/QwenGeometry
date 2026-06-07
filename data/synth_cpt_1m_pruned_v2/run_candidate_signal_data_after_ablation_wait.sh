#!/usr/bin/env bash
set -euo pipefail

WORK=/root/rivermind-data/qwen_ag_lm
. "$WORK/env.sh" 2>/dev/null || true
. "$WORK/venv/bin/activate" 2>/dev/null || true
cd "$WORK"

ABLATION_TAG=${ABLATION_TAG:-unsolved_factctx_top8_adapter_value_v5_grammar_v1}
OUT_DIR=${OUT_DIR:-outputs/final_eval_imo_ag30_qwen_${ABLATION_TAG}}
EVENTS_DIR=${EVENTS_DIR:-$OUT_DIR/events}
SUMMARY_JSONL=${SUMMARY_JSONL:-$OUT_DIR/summary.jsonl}
EXPECTED_PROBLEMS=${EXPECTED_PROBLEMS:-16}
WAIT_INTERVAL=${WAIT_INTERVAL:-300}
ALLOW_PARTIAL=${ALLOW_PARTIAL:-0}
MIN_PARTIAL_SUMMARY_ROWS=${MIN_PARTIAL_SUMMARY_ROWS:-1}
MIN_PARTIAL_SFT_SIGNALS=${MIN_PARTIAL_SFT_SIGNALS:-1}
MIN_PARTIAL_HARD_NEGATIVE_SIGNALS=${MIN_PARTIAL_HARD_NEGATIVE_SIGNALS:-1}
DATA_TAG=${DATA_TAG:-candidate_signals_${ABLATION_TAG}}
DATA_DIR=${DATA_DIR:-data/staged_1m_pruned_v2/${DATA_TAG}}
TRAIN_FILE=$DATA_DIR/candidate_signal_aux_train.jsonl
EVAL_FILE=$DATA_DIR/candidate_signal_aux_eval.jsonl
SUMMARY_FILE=$DATA_DIR/summary.json
HARD_NEG_TRAIN_FILE=$DATA_DIR/candidate_hard_negative_aux_train.jsonl
HARD_NEG_EVAL_FILE=$DATA_DIR/candidate_hard_negative_aux_eval.jsonl
HARD_NEG_SUMMARY_FILE=$DATA_DIR/hard_negative_summary.json
LOG=${LOG:-outputs/${DATA_TAG}.wait_then_build.log}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "$DATA_DIR"
export ABLATION_TAG OUT_DIR EVENTS_DIR SUMMARY_JSONL EXPECTED_PROBLEMS DATA_TAG DATA_DIR
export ALLOW_PARTIAL MIN_PARTIAL_SUMMARY_ROWS MIN_PARTIAL_SFT_SIGNALS MIN_PARTIAL_HARD_NEGATIVE_SIGNALS
export TRAIN_FILE EVAL_FILE SUMMARY_FILE HARD_NEG_TRAIN_FILE HARD_NEG_EVAL_FILE HARD_NEG_SUMMARY_FILE

if [ "$DRY_RUN" = "1" ]; then
  python - <<'PY'
import json
import os
print(json.dumps({
    'ablation_tag': os.environ.get('ABLATION_TAG'),
    'out_dir': os.environ.get('OUT_DIR'),
    'events_dir': os.environ.get('EVENTS_DIR'),
    'summary_jsonl': os.environ.get('SUMMARY_JSONL'),
    'expected_problems': int(os.environ.get('EXPECTED_PROBLEMS', '16')),
    'allow_partial': os.environ.get('ALLOW_PARTIAL') == '1',
    'min_partial_summary_rows': int(os.environ.get('MIN_PARTIAL_SUMMARY_ROWS', '1')),
    'min_partial_sft_signals': int(os.environ.get('MIN_PARTIAL_SFT_SIGNALS', '1')),
    'min_partial_hard_negative_signals': int(os.environ.get('MIN_PARTIAL_HARD_NEGATIVE_SIGNALS', '1')),
    'data_dir': os.environ.get('DATA_DIR'),
    'hard_negative_train_file': os.environ.get('HARD_NEG_TRAIN_FILE'),
}, ensure_ascii=False, indent=2))
PY
  exit 0
fi

echo "waiting for ablation summary: $SUMMARY_JSONL" | tee -a "$LOG"
while true; do
  if [ -s "$SUMMARY_JSONL" ]; then
    rows=$(python - "$SUMMARY_JSONL" <<'PY'
import sys
from pathlib import Path
print(sum(1 for line in Path(sys.argv[1]).read_text(encoding='utf-8', errors='replace').splitlines() if line.strip()))
PY
)
  else
    rows=0
  fi
  read -r sft_signals hard_negative_signals < <(python - "$EVENTS_DIR" <<'PY'
import json
import sys
from pathlib import Path

events_dir = Path(sys.argv[1])
sft = 0
hard = 0
if events_dir.exists():
  for path in sorted(events_dir.glob('*.jsonl')):
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
      if not line.strip():
        continue
      try:
        event = json.loads(line.lstrip('\ufeff'))
      except json.JSONDecodeError:
        continue
      kind = event.get('kind')
      if kind == 'candidate_sft_signal':
        sft += 1
      elif kind == 'candidate_hard_negative_signal':
        hard += 1
print(sft, hard)
PY
)
  date '+%F %T %z' | tee -a "$LOG"
  echo "summary rows: $rows / $EXPECTED_PROBLEMS" | tee -a "$LOG"
  echo "candidate signals: sft=$sft_signals hard_negative=$hard_negative_signals" | tee -a "$LOG"
  if [ "$rows" -ge "$EXPECTED_PROBLEMS" ]; then
    break
  fi
  if [ "$ALLOW_PARTIAL" = "1" ] \
      && [ "$rows" -ge "$MIN_PARTIAL_SUMMARY_ROWS" ] \
      && [ "$sft_signals" -ge "$MIN_PARTIAL_SFT_SIGNALS" ] \
      && [ "$hard_negative_signals" -ge "$MIN_PARTIAL_HARD_NEGATIVE_SIGNALS" ]; then
    echo "partial build enabled; thresholds satisfied before full summary completion" | tee -a "$LOG"
    break
  fi
  sleep "$WAIT_INTERVAL"
done

python -u scripts/build_aux_sft_from_candidate_signals.py \
  --events_dir "$EVENTS_DIR" \
  --train_file "$TRAIN_FILE" \
  --eval_file "$EVAL_FILE" \
  --summary_file "$SUMMARY_FILE" \
  > "$DATA_DIR/build.log" 2>&1

python -u scripts/build_aux_hard_negative_from_candidate_signals.py \
  --events_dir "$EVENTS_DIR" \
  --train_file "$HARD_NEG_TRAIN_FILE" \
  --eval_file "$HARD_NEG_EVAL_FILE" \
  --summary_file "$HARD_NEG_SUMMARY_FILE" \
  > "$DATA_DIR/build_hard_negative.log" 2>&1

cat "$SUMMARY_FILE"
cat "$HARD_NEG_SUMMARY_FILE"
