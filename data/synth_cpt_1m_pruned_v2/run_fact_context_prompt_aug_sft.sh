#!/usr/bin/env bash
set -euo pipefail

WORK=/root/rivermind-data/qwen_ag_lm
. "$WORK/env.sh" 2>/dev/null || true
. "$WORK/venv/bin/activate" 2>/dev/null || true
cd "$WORK"

STAGED=${STAGED:-data/staged_1m_pruned_v2}
FACT_TAG=${FACT_TAG:-factctx_promptaug_top8_stage2max2000_v1}
FACT_DIR=${FACT_DIR:-$STAGED/$FACT_TAG}
AUG_TRAIN=$FACT_DIR/fact_context_prompt_aug_train.jsonl
AUG_EVAL=$FACT_DIR/fact_context_prompt_aug_eval.jsonl
AUG_SUMMARY=$FACT_DIR/prompt_aug_summary.json
MIXED_TRAIN=$FACT_DIR/fact_context_mixed_train.jsonl
MIXED_EVAL=$FACT_DIR/fact_context_mixed_eval.jsonl
SUMMARY=$FACT_DIR/summary.json
LOG=${LOG:-outputs/${FACT_TAG}.prompt_aug_train.log}
mkdir -p "$FACT_DIR" "$(dirname "$LOG")"

BASE_ADAPTER=${BASE_ADAPTER:-outputs/stage2_aux_after_cpt1m_lora_qwen2_5_7b_v1}
OUT=${OUT:-outputs/stage2_fact_context_after_stage2_lora_qwen2_5_7b_${FACT_TAG}}
OLD_TRAIN=${OLD_TRAIN:-$STAGED/stage2_aux_sft_train.jsonl}
OLD_EVAL=${OLD_EVAL:-$STAGED/stage2_aux_sft_eval.jsonl}
JGEX_FACT_TAG=${JGEX_FACT_TAG:-factctx_top8_after_v3_v1}
JGEX_FACT_DIR=${JGEX_FACT_DIR:-$STAGED/$JGEX_FACT_TAG}
JGEX_FACT_TRAIN=${JGEX_FACT_TRAIN:-$JGEX_FACT_DIR/fact_context_aux_train.jsonl}
JGEX_FACT_EVAL=${JGEX_FACT_EVAL:-$JGEX_FACT_DIR/fact_context_aux_eval.jsonl}

FACT_CONTEXT_TOP_K=${FACT_CONTEXT_TOP_K:-8}
FACT_CONTEXT_MAX_LEVEL=${FACT_CONTEXT_MAX_LEVEL:-4}
FACT_CONTEXT_DDAR_TIMEOUT=${FACT_CONTEXT_DDAR_TIMEOUT:-10}
PROMPT_AUG_MAX_ROWS=${PROMPT_AUG_MAX_ROWS:-2000}
OLD_FORMAT_MAX_ROWS=${OLD_FORMAT_MAX_ROWS:-3000}
OLD_EVAL_MAX_ROWS=${OLD_EVAL_MAX_ROWS:-200}
WAIT_INTERVAL=${WAIT_INTERVAL:-300}
DRY_RUN=${DRY_RUN:-0}

export FACT_TAG FACT_DIR AUG_TRAIN AUG_EVAL AUG_SUMMARY MIXED_TRAIN MIXED_EVAL SUMMARY
export BASE_ADAPTER OUT OLD_TRAIN OLD_EVAL JGEX_FACT_TRAIN JGEX_FACT_EVAL
export FACT_CONTEXT_TOP_K FACT_CONTEXT_MAX_LEVEL FACT_CONTEXT_DDAR_TIMEOUT
export PROMPT_AUG_MAX_ROWS OLD_FORMAT_MAX_ROWS OLD_EVAL_MAX_ROWS

if [ "$DRY_RUN" = "1" ]; then
  python - <<'PY'
import json
import os
print(json.dumps({
    'fact_tag': os.environ.get('FACT_TAG'),
    'fact_dir': os.environ.get('FACT_DIR'),
    'base_adapter': os.environ.get('BASE_ADAPTER'),
    'out': os.environ.get('OUT'),
    'prompt_aug_max_rows': int(os.environ.get('PROMPT_AUG_MAX_ROWS', '2000')),
    'fact_context_top_k': int(os.environ.get('FACT_CONTEXT_TOP_K', '8')),
    'fact_context_max_level': int(os.environ.get('FACT_CONTEXT_MAX_LEVEL', '4')),
    'fact_context_ddar_timeout': int(os.environ.get('FACT_CONTEXT_DDAR_TIMEOUT', '10')),
}, ensure_ascii=False, indent=2))
PY
  exit 0
fi

echo "building prompt-augmented fact-context rows: $FACT_TAG" | tee -a "$LOG"
if [ ! -s "$AUG_SUMMARY" ]; then
  xvfb-run -a -s "-screen 0 1024x768x24" python -u scripts/build_fact_context_from_aux_sft.py \
    --ag_repo repos/alphageometry \
    --defs_file repos/alphageometry/defs.txt \
    --rules_file repos/alphageometry/rules.txt \
    --input_file "$OLD_TRAIN" \
    --input_file "$OLD_EVAL" \
    --train_file "$AUG_TRAIN" \
    --eval_file "$AUG_EVAL" \
    --summary_file "$AUG_SUMMARY" \
    --fact_context_top_k "$FACT_CONTEXT_TOP_K" \
    --fact_context_max_level "$FACT_CONTEXT_MAX_LEVEL" \
    --fact_context_ddar_timeout "$FACT_CONTEXT_DDAR_TIMEOUT" \
    --max_rows "$PROMPT_AUG_MAX_ROWS" \
    > "$FACT_DIR/build_prompt_aug.log" 2>&1
fi
cat "$AUG_SUMMARY" | tee -a "$LOG"

python - "$AUG_TRAIN" "$AUG_EVAL" "$JGEX_FACT_TRAIN" "$JGEX_FACT_EVAL" "$OLD_TRAIN" "$OLD_EVAL" "$MIXED_TRAIN" "$MIXED_EVAL" "$SUMMARY" <<'PY'
import json
import os
import sys
from pathlib import Path

aug_train, aug_eval, jgex_train, jgex_eval, old_train, old_eval, mixed_train, mixed_eval, summary_path = map(Path, sys.argv[1:])
old_limit = int(os.environ.get('OLD_FORMAT_MAX_ROWS', '3000'))
old_eval_limit = int(os.environ.get('OLD_EVAL_MAX_ROWS', '200'))

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

aug_train_rows = read_jsonl(aug_train)
aug_eval_rows = read_jsonl(aug_eval)
jgex_train_rows = read_jsonl(jgex_train)
jgex_eval_rows = read_jsonl(jgex_eval)
old_train_rows = read_jsonl(old_train, old_limit)
old_eval_rows = read_jsonl(old_eval, old_eval_limit)

mixed_train_rows = aug_train_rows + jgex_train_rows + old_train_rows
mixed_eval_rows = aug_eval_rows + jgex_eval_rows + old_eval_rows
write_jsonl(mixed_train, mixed_train_rows)
write_jsonl(mixed_eval, mixed_eval_rows)

summary = {
    'status': 'prepared',
    'aug_train_rows': len(aug_train_rows),
    'aug_eval_rows': len(aug_eval_rows),
    'jgex_train_rows': len(jgex_train_rows),
    'jgex_eval_rows': len(jgex_eval_rows),
    'old_train_rows': len(old_train_rows),
    'old_eval_rows': len(old_eval_rows),
    'mixed_train_rows': len(mixed_train_rows),
    'mixed_eval_rows': len(mixed_eval_rows),
    'mixed_train': str(mixed_train),
    'mixed_eval': str(mixed_eval),
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
if len(aug_train_rows) < 100:
    raise SystemExit('too few prompt-augmented fact-context rows')
PY

echo "waiting for current Qwen LoRA training to finish before prompt-aug SFT" | tee -a "$LOG"
while pgrep -f "scripts/train_qwen_aux_lora.py" >/dev/null; do
  date '+%F %T %z' | tee -a "$LOG"
  pgrep -af "scripts/train_qwen_aux_lora.py" | tee -a "$LOG" || true
  sleep "$WAIT_INTERVAL"
done

rm -rf "$OUT"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u scripts/train_qwen_aux_lora.py \
  --model_name_or_path models/Qwen2.5-7B \
  --init_adapter_path "$BASE_ADAPTER" \
  --train_file "$MIXED_TRAIN" \
  --eval_file "$MIXED_EVAL" \
  --output_dir "$OUT" \
  --loss_mode target \
  --max_length 1536 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --logging_steps 10 \
  --eval_steps 50 \
  --save_steps 100 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  > "$OUT/train.log" 2>&1

python - "$SUMMARY" "$OUT" <<'PY'
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
