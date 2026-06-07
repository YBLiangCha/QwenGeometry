#!/usr/bin/env bash
set -euo pipefail

WORK=/root/rivermind-data/qwen_ag_lm
. "$WORK/env.sh" 2>/dev/null || true
. "$WORK/venv/bin/activate" 2>/dev/null || true
cd "$WORK"

STAGED=data/staged_1m_pruned_v2
FINAL_PATH_FILE=outputs/final_adapter_path.txt

while [ ! -s "$FINAL_PATH_FILE" ]; do
  sleep 300
done

FINAL_ADAPTER=$(cat "$FINAL_PATH_FILE")
while [ ! -s "$FINAL_ADAPTER/adapter_model.safetensors" ]; do
  sleep 120
done

HEALTH_DIR=outputs/final_adapter_health_v1
mkdir -p "$HEALTH_DIR"
TRAIN_LOG="$FINAL_ADAPTER/train.log"
export CUDA_VISIBLE_DEVICES=0
python -u scripts/check_adapter_health.py \
  --model_name_or_path models/Qwen2.5-7B \
  --adapter_path "$FINAL_ADAPTER" \
  --eval_file "$STAGED/stage2_aux_sft_eval.jsonl" \
  --train_log "$TRAIN_LOG" \
  --out_file "$HEALTH_DIR/sample_outputs.jsonl" \
  --summary_file "$HEALTH_DIR/summary.json" \
  --sample_count 16 \
  --min_format_ratio 0.75 \
  > "$HEALTH_DIR/health.log" 2>&1

BASELINE_TAG=${BASELINE_TAG:-v1}
DDAR_OUT=outputs/final_eval_imo_ag30_ddar_${BASELINE_TAG}
QWEN_OUT=outputs/final_eval_imo_ag30_qwen_${BASELINE_TAG}
COMPARISON_OUT=outputs/final_baseline_comparison_${BASELINE_TAG}.json
QWEN_CANDIDATE_DDAR_WORKERS=${QWEN_CANDIDATE_DDAR_WORKERS:-4}
QWEN_CANDIDATE_WALL_TIMEOUT=${QWEN_CANDIDATE_WALL_TIMEOUT:-180}
QWEN_CANDIDATE_EVAL_LIMIT=${QWEN_CANDIDATE_EVAL_LIMIT:-0}
QWEN_CANDIDATE_DEPTH_EVAL_LIMIT=${QWEN_CANDIDATE_DEPTH_EVAL_LIMIT:-0}
QWEN_CANDIDATE_QUALITY_MULTIPLIER=${QWEN_CANDIDATE_QUALITY_MULTIPLIER:-2}
QWEN_CANDIDATE_DSL_FILTER=${QWEN_CANDIDATE_DSL_FILTER:-0}
QWEN_CANDIDATE_DSL_TOKEN_MASK=${QWEN_CANDIDATE_DSL_TOKEN_MASK:-0}
QWEN_CANDIDATE_POINT_REPAIR=${QWEN_CANDIDATE_POINT_REPAIR:-0}
QWEN_CANDIDATE_PROMPT_SAMPLING=${QWEN_CANDIDATE_PROMPT_SAMPLING:-none}
QWEN_CANDIDATE_TEMPLATE_BACKFILL=${QWEN_CANDIDATE_TEMPLATE_BACKFILL:-0}
QWEN_CANDIDATE_RERANK=${QWEN_CANDIDATE_RERANK:-none}
QWEN_CANDIDATE_VALUE_MODEL=${QWEN_CANDIDATE_VALUE_MODEL:-}
QWEN_LM_FACT_CONTEXT_TOP_K=${QWEN_LM_FACT_CONTEXT_TOP_K:-0}
export DDAR_OUT QWEN_OUT COMPARISON_OUT QWEN_CANDIDATE_DDAR_WORKERS QWEN_CANDIDATE_WALL_TIMEOUT QWEN_CANDIDATE_EVAL_LIMIT QWEN_CANDIDATE_DEPTH_EVAL_LIMIT QWEN_CANDIDATE_QUALITY_MULTIPLIER QWEN_CANDIDATE_DSL_FILTER QWEN_CANDIDATE_DSL_TOKEN_MASK QWEN_CANDIDATE_POINT_REPAIR QWEN_CANDIDATE_PROMPT_SAMPLING QWEN_CANDIDATE_TEMPLATE_BACKFILL QWEN_CANDIDATE_RERANK QWEN_CANDIDATE_VALUE_MODEL QWEN_LM_FACT_CONTEXT_TOP_K
mkdir -p "$DDAR_OUT" "$QWEN_OUT"

if [ "$QWEN_CANDIDATE_DSL_FILTER" = "1" ]; then
  QWEN_DSL_FILTER_FLAG=--candidate_dsl_filter
else
  QWEN_DSL_FILTER_FLAG=--no-candidate_dsl_filter
fi
if [ "$QWEN_CANDIDATE_DSL_TOKEN_MASK" = "1" ]; then
  QWEN_DSL_TOKEN_MASK_FLAG=--candidate_dsl_token_mask
else
  QWEN_DSL_TOKEN_MASK_FLAG=--no-candidate_dsl_token_mask
fi
if [ "$QWEN_CANDIDATE_POINT_REPAIR" = "1" ]; then
  QWEN_POINT_REPAIR_FLAG=--candidate_point_repair
else
  QWEN_POINT_REPAIR_FLAG=--no-candidate_point_repair
fi
if [ "$QWEN_CANDIDATE_TEMPLATE_BACKFILL" = "1" ]; then
  QWEN_TEMPLATE_BACKFILL_FLAG=--candidate_template_backfill
else
  QWEN_TEMPLATE_BACKFILL_FLAG=--no-candidate_template_backfill
fi
VALUE_MODEL_ARGS=()
if [ -n "$QWEN_CANDIDATE_VALUE_MODEL" ]; then
  VALUE_MODEL_ARGS=(--candidate_value_model "$QWEN_CANDIDATE_VALUE_MODEL")
fi

if [ ! -s "$DDAR_OUT/summary.json" ]; then
  xvfb-run -a -s "-screen 0 1024x768x24" python -u scripts/run_qwen_ag_benchmark.py \
    --script_dir scripts \
    --ag_repo repos/alphageometry \
    --problems_file repos/alphageometry/imo_ag_30.txt \
    --defs_file repos/alphageometry/defs.txt \
    --rules_file repos/alphageometry/rules.txt \
    --out_dir "$DDAR_OUT" \
    --mode ddar \
    --max_level 1000 \
    --ddar_timeout 600 \
    > "$DDAR_OUT/run.log" 2>&1
fi

if [ ! -s "$QWEN_OUT/summary.json" ]; then
  xvfb-run -a -s "-screen 0 1024x768x24" python -u scripts/run_qwen_ag_benchmark.py \
    --script_dir scripts \
    --ag_repo repos/alphageometry \
    --problems_file repos/alphageometry/imo_ag_30.txt \
    --defs_file repos/alphageometry/defs.txt \
    --rules_file repos/alphageometry/rules.txt \
    --out_dir "$QWEN_OUT" \
    --mode qwen \
    --qwen_model models/Qwen2.5-7B \
    --adapter_path "$FINAL_ADAPTER" \
    --dtype bf16 \
    --device_map cuda:0 \
    --root_max_level 1000 \
    --root_ddar_timeout 600 \
    --candidate_max_level 300 \
    --candidate_ddar_timeout 180 \
    --candidate_wall_timeout "$QWEN_CANDIDATE_WALL_TIMEOUT" \
    --candidate_eval_limit "$QWEN_CANDIDATE_EVAL_LIMIT" \
    --candidate_depth_eval_limit "$QWEN_CANDIDATE_DEPTH_EVAL_LIMIT" \
    --beam_size 8 \
    --search_depth 2 \
    --num_return_sequences 8 \
    --max_new_tokens 64 \
    --temperature 0.7 \
    --top_p 0.95 \
    --candidate_quality_multiplier "$QWEN_CANDIDATE_QUALITY_MULTIPLIER" \
    "$QWEN_DSL_FILTER_FLAG" \
    "$QWEN_DSL_TOKEN_MASK_FLAG" \
    "$QWEN_POINT_REPAIR_FLAG" \
    --candidate_point_mask \
    --candidate_canonical_dedup \
    --candidate_prompt_sampling "$QWEN_CANDIDATE_PROMPT_SAMPLING" \
    "$QWEN_TEMPLATE_BACKFILL_FLAG" \
    --candidate_rerank "$QWEN_CANDIDATE_RERANK" \
    "${VALUE_MODEL_ARGS[@]}" \
    --candidate_ddar_workers "$QWEN_CANDIDATE_DDAR_WORKERS" \
    --lm_fact_context_top_k "$QWEN_LM_FACT_CONTEXT_TOP_K" \
    > "$QWEN_OUT/run.log" 2>&1
fi

python - <<'PY' > "$COMPARISON_OUT"
import json
import os
from pathlib import Path

def load_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}

def load_rows(path):
    p = Path(path)
    rows = []
    if not p.exists():
        return rows
    for line in p.read_text(encoding='utf-8', errors='replace').splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows

ddar_out = os.environ['DDAR_OUT']
qwen_out = os.environ['QWEN_OUT']
ddar = load_json(f'{ddar_out}/summary.json')
qwen = load_json(f'{qwen_out}/summary.json')
health = load_json('outputs/final_adapter_health_v1/summary.json')
ddar_rows = load_rows(f'{ddar_out}/summary.jsonl')
qwen_rows = load_rows(f'{qwen_out}/summary.jsonl')

def solved_names(rows):
    return sorted(row.get('name') or row.get('problem') for row in rows if row.get('solved'))

def error_rows(rows):
    return [
        {'name': row.get('name') or row.get('problem'), 'error': row.get('error')}
        for row in rows
        if row.get('error')
    ]

ddar_solved_names = solved_names(ddar_rows)
qwen_solved_names = solved_names(qwen_rows)
ddar_set = set(ddar_solved_names)
qwen_set = set(qwen_solved_names)
final_adapter = Path('outputs/final_adapter_path.txt').read_text().strip()
comparison = {
    'final_adapter': final_adapter,
    'health_passed': health.get('passed'),
    'health_format_ratio': health.get('format_ratio'),
    'health_samples': health.get('samples'),
    'ddar_out': ddar_out,
    'qwen_out': qwen_out,
    'qwen_candidate_ddar_workers': int(os.environ.get('QWEN_CANDIDATE_DDAR_WORKERS', '1')),
    'qwen_candidate_wall_timeout': int(os.environ.get('QWEN_CANDIDATE_WALL_TIMEOUT', '180')),
    'qwen_candidate_eval_limit': int(os.environ.get('QWEN_CANDIDATE_EVAL_LIMIT', '0')),
    'qwen_candidate_quality_multiplier': int(os.environ.get('QWEN_CANDIDATE_QUALITY_MULTIPLIER', '1')),
    'qwen_candidate_dsl_filter': os.environ.get('QWEN_CANDIDATE_DSL_FILTER', '0') == '1',
    'qwen_candidate_dsl_token_mask': os.environ.get('QWEN_CANDIDATE_DSL_TOKEN_MASK', '0') == '1',
    'qwen_candidate_point_repair': os.environ.get('QWEN_CANDIDATE_POINT_REPAIR', '0') == '1',
    'qwen_candidate_prompt_sampling': os.environ.get('QWEN_CANDIDATE_PROMPT_SAMPLING', 'none'),
    'qwen_candidate_template_backfill': os.environ.get('QWEN_CANDIDATE_TEMPLATE_BACKFILL', '0') == '1',
    'qwen_candidate_rerank': os.environ.get('QWEN_CANDIDATE_RERANK', 'none'),
    'qwen_candidate_value_model': os.environ.get('QWEN_CANDIDATE_VALUE_MODEL') or None,
    'ddar_solved': ddar.get('solved'),
    'qwen_solved': qwen.get('solved'),
    'num_problems': qwen.get('num_problems') or ddar.get('num_problems'),
    'ddar_solved_names': ddar_solved_names,
    'qwen_solved_names': qwen_solved_names,
    'qwen_only_solved': sorted(qwen_set - ddar_set),
    'ddar_only_solved': sorted(ddar_set - qwen_set),
    'qwen_errors': error_rows(qwen_rows),
    'ddar_errors': error_rows(ddar_rows),
}
if comparison['ddar_solved'] is not None and comparison['qwen_solved'] is not None:
    comparison['absolute_gain'] = comparison['qwen_solved'] - comparison['ddar_solved']
print(json.dumps(comparison, ensure_ascii=False, indent=2))
PY
