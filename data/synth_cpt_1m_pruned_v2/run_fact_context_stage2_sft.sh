#!/usr/bin/env bash
set -euo pipefail

WORK=/root/rivermind-data/qwen_ag_lm
. "$WORK/env.sh" 2>/dev/null || true
. "$WORK/venv/bin/activate" 2>/dev/null || true
cd "$WORK"

STAGED=${STAGED:-data/staged_1m_pruned_v2}
FACT_TAG=${FACT_TAG:-factctx_top8_v1}
FACT_DIR=${FACT_DIR:-$STAGED/$FACT_TAG}
FACT_RAW=$FACT_DIR/fact_context_aux_raw.jsonl
FACT_TRAIN=$FACT_DIR/fact_context_aux_train.jsonl
FACT_EVAL=$FACT_DIR/fact_context_aux_eval.jsonl
MIXED_TRAIN=$FACT_DIR/fact_context_mixed_train.jsonl
MIXED_EVAL=$FACT_DIR/fact_context_mixed_eval.jsonl
SUMMARY=$FACT_DIR/summary.json
mkdir -p "$FACT_DIR"

BASE_ADAPTER=${BASE_ADAPTER:-outputs/stage2_aux_after_cpt1m_lora_qwen2_5_7b_v1}
OUT=${OUT:-outputs/stage2_fact_context_after_stage2_lora_qwen2_5_7b_${FACT_TAG}}
PROBLEMS_FILE=${PROBLEMS_FILE:-repos/alphageometry/jgex_ag_231.txt}
MAX_PROBLEMS=${MAX_PROBLEMS:-}
FACT_CONTEXT_TOP_K=${FACT_CONTEXT_TOP_K:-8}
FACT_CONTEXT_MAX_LEVEL=${FACT_CONTEXT_MAX_LEVEL:-4}
FACT_CONTEXT_DDAR_TIMEOUT=${FACT_CONTEXT_DDAR_TIMEOUT:-30}
OLD_FORMAT_MAX_ROWS=${OLD_FORMAT_MAX_ROWS:-4000}
EVAL_MOD=${EVAL_MOD:-10}
PROMOTE=${PROMOTE:-0}
DRY_RUN=${DRY_RUN:-0}
export STAGED FACT_TAG FACT_DIR BASE_ADAPTER OUT PROBLEMS_FILE FACT_CONTEXT_TOP_K
export FACT_CONTEXT_MAX_LEVEL FACT_CONTEXT_DDAR_TIMEOUT OLD_FORMAT_MAX_ROWS PROMOTE

MAX_PROBLEM_ARGS=()
if [ -n "$MAX_PROBLEMS" ]; then
  MAX_PROBLEM_ARGS=(--max_problems "$MAX_PROBLEMS")
fi

if [ "$DRY_RUN" = "1" ]; then
  python - <<'PY'
import json, os
print(json.dumps({
    'fact_dir': os.environ.get('FACT_DIR'),
    'base_adapter': os.environ.get('BASE_ADAPTER'),
    'out': os.environ.get('OUT'),
    'problems_file': os.environ.get('PROBLEMS_FILE'),
    'fact_context_top_k': int(os.environ.get('FACT_CONTEXT_TOP_K', '8')),
    'fact_context_max_level': int(os.environ.get('FACT_CONTEXT_MAX_LEVEL', '4')),
    'fact_context_ddar_timeout': int(os.environ.get('FACT_CONTEXT_DDAR_TIMEOUT', '30')),
    'old_format_max_rows': int(os.environ.get('OLD_FORMAT_MAX_ROWS', '4000')),
    'promote': os.environ.get('PROMOTE', '0') == '1',
}, ensure_ascii=False, indent=2))
PY
  exit 0
fi

if [ ! -s "$FACT_RAW" ]; then
  xvfb-run -a -s "-screen 0 1024x768x24" python -u scripts/make_aux_sft_from_ag_file.py \
    --ag_repo repos/alphageometry \
    --problems_file "$PROBLEMS_FILE" \
    --defs_file repos/alphageometry/defs.txt \
    --rules_file repos/alphageometry/rules.txt \
    --out_file "$FACT_RAW" \
    "${MAX_PROBLEM_ARGS[@]}" \
    --min_prefix_clauses 1 \
    --fact_context_top_k "$FACT_CONTEXT_TOP_K" \
    --fact_context_max_level "$FACT_CONTEXT_MAX_LEVEL" \
    --fact_context_ddar_timeout "$FACT_CONTEXT_DDAR_TIMEOUT" \
    > "$FACT_DIR/build_fact_context.log" 2>&1
fi

python - "$FACT_RAW" "$FACT_TRAIN" "$FACT_EVAL" "$MIXED_TRAIN" "$MIXED_EVAL" "$SUMMARY" <<'PY'
import json
import os
import sys
from pathlib import Path

raw, fact_train, fact_eval, mixed_train, mixed_eval, summary = map(Path, sys.argv[1:])
eval_mod = int(os.environ.get('EVAL_MOD', '10'))
old_limit = int(os.environ.get('OLD_FORMAT_MAX_ROWS', '4000'))
old_train = Path(os.environ.get('OLD_TRAIN', 'data/staged_1m_pruned_v2/stage2_aux_sft_train.jsonl'))
old_eval = Path(os.environ.get('OLD_EVAL', 'data/staged_1m_pruned_v2/stage2_aux_sft_eval.jsonl'))

fact_rows = []
for line in raw.read_text(encoding='utf-8', errors='replace').splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if '{D}' not in row.get('prompt', '') or not row.get('fact_context'):
        continue
    fact_rows.append(row)

def split(row):
    key = row.get('source_problem') or row.get('id') or ''
    return 'eval' if eval_mod > 0 and sum(key.encode('utf-8')) % eval_mod == 0 else 'train'

train_rows = [row for row in fact_rows if split(row) == 'train']
eval_rows = [row for row in fact_rows if split(row) == 'eval']
if not eval_rows and len(train_rows) > 5:
    eval_rows = train_rows[-max(1, len(train_rows) // 10):]
    train_rows = train_rows[:-len(eval_rows)]

def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
        encoding='utf-8',
    )

write(fact_train, train_rows)
write(fact_eval, eval_rows)

mixed_train_rows = list(train_rows)
if old_limit > 0 and old_train.exists():
    for i, line in enumerate(old_train.read_text(encoding='utf-8', errors='replace').splitlines()):
        if i >= old_limit:
            break
        if line.strip():
            mixed_train_rows.append(json.loads(line))

mixed_eval_rows = list(eval_rows)
if old_eval.exists():
    for line in old_eval.read_text(encoding='utf-8', errors='replace').splitlines():
        if line.strip():
            mixed_eval_rows.append(json.loads(line))

write(mixed_train, mixed_train_rows)
write(mixed_eval, mixed_eval_rows)

summary_obj = {
    'fact_raw_rows': len(fact_rows),
    'fact_train_rows': len(train_rows),
    'fact_eval_rows': len(eval_rows),
    'mixed_train_rows': len(mixed_train_rows),
    'mixed_eval_rows': len(mixed_eval_rows),
    'old_format_max_rows': old_limit,
    'fact_train': str(fact_train),
    'fact_eval': str(fact_eval),
    'mixed_train': str(mixed_train),
    'mixed_eval': str(mixed_eval),
}
summary.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary_obj, ensure_ascii=False, indent=2))
if len(train_rows) < 16:
    raise SystemExit('too few fact-context train rows; increase source data or relax mining settings')
PY

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

if [ "$PROMOTE" = "1" ]; then
  echo "$OUT" > outputs/final_adapter_path.txt
fi

cat "$SUMMARY"
