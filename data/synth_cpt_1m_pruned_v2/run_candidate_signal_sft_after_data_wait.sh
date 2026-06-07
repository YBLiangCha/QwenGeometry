#!/usr/bin/env bash
set -euo pipefail

WORK=/root/rivermind-data/qwen_ag_lm
. "$WORK/env.sh" 2>/dev/null || true
. "$WORK/venv/bin/activate" 2>/dev/null || true
cd "$WORK"

ABLATION_TAG=${ABLATION_TAG:-unsolved_factctx_top8_adapter_value_v5_grammar_v1}
DATA_TAG=${DATA_TAG:-candidate_signals_${ABLATION_TAG}}
SFT_TAG=${SFT_TAG:-candidate_signal_sft_${ABLATION_TAG}}
STAGED=${STAGED:-data/staged_1m_pruned_v2}
DATA_DIR=${DATA_DIR:-$STAGED/$DATA_TAG}
SIGNAL_TRAIN=${SIGNAL_TRAIN:-$DATA_DIR/candidate_signal_aux_train.jsonl}
SIGNAL_EVAL=${SIGNAL_EVAL:-$DATA_DIR/candidate_signal_aux_eval.jsonl}
SIGNAL_SUMMARY=${SIGNAL_SUMMARY:-$DATA_DIR/summary.json}
HARD_NEG_TRAIN=${HARD_NEG_TRAIN:-$DATA_DIR/candidate_hard_negative_aux_train.jsonl}
HARD_NEG_EVAL=${HARD_NEG_EVAL:-$DATA_DIR/candidate_hard_negative_aux_eval.jsonl}
HARD_NEG_SUMMARY=${HARD_NEG_SUMMARY:-$DATA_DIR/hard_negative_summary.json}

FACT_TAG=${FACT_TAG:-factctx_top8_after_v3_v1}
FACT_DIR=${FACT_DIR:-$STAGED/$FACT_TAG}
FACT_MIX_TRAIN=${FACT_MIX_TRAIN:-$FACT_DIR/fact_context_mixed_train.jsonl}
FACT_MIX_EVAL=${FACT_MIX_EVAL:-$FACT_DIR/fact_context_mixed_eval.jsonl}
BASE_ADAPTER=${BASE_ADAPTER:-outputs/stage2_fact_context_after_stage2_lora_qwen2_5_7b_${FACT_TAG}}

OUT=${OUT:-outputs/stage3_candidate_signal_after_factctx_lora_qwen2_5_7b_${SFT_TAG}}
WORK_DIR=${WORK_DIR:-$STAGED/$SFT_TAG}
MIXED_TRAIN=${MIXED_TRAIN:-$WORK_DIR/candidate_signal_mixed_train.jsonl}
MIXED_EVAL=${MIXED_EVAL:-$WORK_DIR/candidate_signal_mixed_eval.jsonl}
RUN_SUMMARY=${RUN_SUMMARY:-$WORK_DIR/summary.json}
LOG=${LOG:-outputs/${SFT_TAG}.wait_then_train.log}

WAIT_INTERVAL=${WAIT_INTERVAL:-300}
MIN_TRAIN_ROWS=${MIN_TRAIN_ROWS:-1}
MAX_FACT_MIX_ROWS=${MAX_FACT_MIX_ROWS:-2000}
MAX_FACT_EVAL_ROWS=${MAX_FACT_EVAL_ROWS:-400}
USE_HARD_NEGATIVES=${USE_HARD_NEGATIVES:-1}
UNLIKELIHOOD_WEIGHT=${UNLIKELIHOOD_WEIGHT:-0.1}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "$WORK_DIR" "$(dirname "$LOG")"
export ABLATION_TAG DATA_TAG SFT_TAG DATA_DIR SIGNAL_TRAIN SIGNAL_EVAL SIGNAL_SUMMARY
export HARD_NEG_TRAIN HARD_NEG_EVAL HARD_NEG_SUMMARY USE_HARD_NEGATIVES UNLIKELIHOOD_WEIGHT
export FACT_TAG FACT_DIR FACT_MIX_TRAIN FACT_MIX_EVAL BASE_ADAPTER OUT WORK_DIR
export MIXED_TRAIN MIXED_EVAL RUN_SUMMARY MIN_TRAIN_ROWS MAX_FACT_MIX_ROWS MAX_FACT_EVAL_ROWS

if [ "$DRY_RUN" = "1" ]; then
  python - <<'PY'
import json
import os
print(json.dumps({
    'ablation_tag': os.environ.get('ABLATION_TAG'),
    'data_dir': os.environ.get('DATA_DIR'),
    'signal_train': os.environ.get('SIGNAL_TRAIN'),
    'signal_summary': os.environ.get('SIGNAL_SUMMARY'),
    'hard_neg_train': os.environ.get('HARD_NEG_TRAIN'),
    'hard_neg_summary': os.environ.get('HARD_NEG_SUMMARY'),
    'use_hard_negatives': os.environ.get('USE_HARD_NEGATIVES'),
    'unlikelihood_weight': float(os.environ.get('UNLIKELIHOOD_WEIGHT', '0.1')),
    'base_adapter': os.environ.get('BASE_ADAPTER'),
    'out': os.environ.get('OUT'),
    'mixed_train': os.environ.get('MIXED_TRAIN'),
    'mixed_eval': os.environ.get('MIXED_EVAL'),
    'min_train_rows': int(os.environ.get('MIN_TRAIN_ROWS', '1')),
    'max_fact_mix_rows': int(os.environ.get('MAX_FACT_MIX_ROWS', '2000')),
}, ensure_ascii=False, indent=2))
PY
  exit 0
fi

echo "waiting for fact-context adapter: $BASE_ADAPTER/adapter_model.safetensors" | tee -a "$LOG"
while [ ! -s "$BASE_ADAPTER/adapter_model.safetensors" ]; do
  date '+%F %T %z' | tee -a "$LOG"
  sleep "$WAIT_INTERVAL"
done

echo "waiting for candidate signal summary: $SIGNAL_SUMMARY" | tee -a "$LOG"
while [ ! -s "$SIGNAL_SUMMARY" ]; do
  date '+%F %T %z' | tee -a "$LOG"
  sleep "$WAIT_INTERVAL"
done

if [ "$USE_HARD_NEGATIVES" = "1" ]; then
  echo "waiting for candidate hard-negative summary: $HARD_NEG_SUMMARY" | tee -a "$LOG"
  while [ ! -s "$HARD_NEG_SUMMARY" ]; do
    date '+%F %T %z' | tee -a "$LOG"
    sleep "$WAIT_INTERVAL"
  done
fi

train_rows=$(python - "$SIGNAL_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(int(summary.get('train_rows') or 0))
PY
)
echo "candidate signal train rows: $train_rows" | tee -a "$LOG"

hard_neg_train_rows=0
hard_neg_eval_rows=0
if [ "$USE_HARD_NEGATIVES" = "1" ]; then
  hard_neg_train_rows=$(python - "$HARD_NEG_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(int(summary.get('train_rows') or 0))
PY
)
  hard_neg_eval_rows=$(python - "$HARD_NEG_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(int(summary.get('eval_rows') or 0))
PY
)
fi
echo "candidate hard-negative train/eval rows: $hard_neg_train_rows / $hard_neg_eval_rows" | tee -a "$LOG"

if [ "$train_rows" -lt "$MIN_TRAIN_ROWS" ]; then
  TRAIN_ROWS="$train_rows" python - <<'PY'
import json
import os
from pathlib import Path
summary = {
    'status': 'skipped_too_few_candidate_signal_rows',
    'signal_summary': os.environ.get('SIGNAL_SUMMARY'),
    'train_rows': int(os.environ.get('TRAIN_ROWS', '0') or 0),
    'min_train_rows': int(os.environ.get('MIN_TRAIN_ROWS', '1')),
}
out = Path(os.environ.get('RUN_SUMMARY'))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
  exit 0
fi

python - "$SIGNAL_TRAIN" "$SIGNAL_EVAL" "$FACT_MIX_TRAIN" "$FACT_MIX_EVAL" "$MIXED_TRAIN" "$MIXED_EVAL" "$RUN_SUMMARY" <<'PY'
import json
import os
import sys
from pathlib import Path

signal_train, signal_eval, fact_train, fact_eval, mixed_train, mixed_eval, run_summary = map(Path, sys.argv[1:])
max_fact_train = int(os.environ.get('MAX_FACT_MIX_ROWS', '2000'))
max_fact_eval = int(os.environ.get('MAX_FACT_EVAL_ROWS', '400'))

def read_jsonl(path: Path, limit: int | None = None):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit is not None and len(rows) >= limit:
            break
    return rows

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')

signal_train_rows = read_jsonl(signal_train)
signal_eval_rows = read_jsonl(signal_eval)
fact_train_rows = read_jsonl(fact_train, max_fact_train)
fact_eval_rows = read_jsonl(fact_eval, max_fact_eval)

mixed_train_rows = signal_train_rows + fact_train_rows
mixed_eval_rows = signal_eval_rows + fact_eval_rows
write_jsonl(mixed_train, mixed_train_rows)
write_jsonl(mixed_eval, mixed_eval_rows)

summary = {
    'status': 'prepared',
    'signal_train_rows': len(signal_train_rows),
    'signal_eval_rows': len(signal_eval_rows),
    'hard_negative_train': os.environ.get('HARD_NEG_TRAIN'),
    'hard_negative_eval': os.environ.get('HARD_NEG_EVAL'),
    'hard_negative_summary': os.environ.get('HARD_NEG_SUMMARY'),
    'use_hard_negatives': os.environ.get('USE_HARD_NEGATIVES') == '1',
    'unlikelihood_weight': float(os.environ.get('UNLIKELIHOOD_WEIGHT', '0.1')),
    'fact_train_rows': len(fact_train_rows),
    'fact_eval_rows': len(fact_eval_rows),
    'mixed_train_rows': len(mixed_train_rows),
    'mixed_eval_rows': len(mixed_eval_rows),
    'signal_train': str(signal_train),
    'signal_eval': str(signal_eval),
    'mixed_train': str(mixed_train),
    'mixed_eval': str(mixed_eval),
}
run_summary.parent.mkdir(parents=True, exist_ok=True)
run_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

rm -rf "$OUT"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NEG_ARGS=()
if [ "$USE_HARD_NEGATIVES" = "1" ]; then
  NEG_ARGS=(
    --negative_train_file "$HARD_NEG_TRAIN"
    --negative_eval_file "$HARD_NEG_EVAL"
    --unlikelihood_weight "$UNLIKELIHOOD_WEIGHT"
  )
fi
python -u scripts/train_qwen_aux_lora.py \
  --model_name_or_path models/Qwen2.5-7B \
  --init_adapter_path "$BASE_ADAPTER" \
  --train_file "$MIXED_TRAIN" \
  --eval_file "$MIXED_EVAL" \
  --output_dir "$OUT" \
  --loss_mode target \
  "${NEG_ARGS[@]}" \
  --max_length 1536 \
  --learning_rate 2e-5 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --logging_steps 5 \
  --eval_steps 25 \
  --save_steps 50 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  > "$OUT/train.log" 2>&1

python - "$RUN_SUMMARY" "$OUT" <<'PY'
import json
import sys
from pathlib import Path
summary_path = Path(sys.argv[1])
summary = json.loads(summary_path.read_text(encoding='utf-8'))
summary['status'] = 'trained'
summary['output_dir'] = sys.argv[2]
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
