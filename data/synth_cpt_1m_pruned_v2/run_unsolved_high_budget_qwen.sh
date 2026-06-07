#!/usr/bin/env bash
set -euo pipefail

WORK=/root/rivermind-data/qwen_ag_lm
. "$WORK/env.sh" 2>/dev/null || true
. "$WORK/venv/bin/activate" 2>/dev/null || true
cd "$WORK"

FINAL_PATH_FILE=outputs/final_adapter_path.txt
if [ -n "${FINAL_ADAPTER_OVERRIDE:-}" ]; then
  FINAL_ADAPTER="$FINAL_ADAPTER_OVERRIDE"
else
  FINAL_ADAPTER=$(cat "$FINAL_PATH_FILE")
fi

SOURCE_QWEN_JSONL=${SOURCE_QWEN_JSONL:-outputs/final_eval_imo_ag30_qwen_v1/summary.jsonl}
UNSOLVED_NAMES=${UNSOLVED_NAMES:-}
if [ -z "$UNSOLVED_NAMES" ] && [ -s "$SOURCE_QWEN_JSONL" ]; then
  UNSOLVED_NAMES=$(python - "$SOURCE_QWEN_JSONL" <<'PY'
import json
import sys
from pathlib import Path

rows = []
for line in Path(sys.argv[1]).read_text(encoding='utf-8', errors='replace').splitlines():
    if line.strip():
        rows.append(json.loads(line))
print(','.join(row.get('name') or row.get('problem') for row in rows if not row.get('solved')))
PY
)
fi

UNSOLVED_NAMES=${UNSOLVED_NAMES:-translated_imo_2000_p6,translated_imo_2004_p1,translated_imo_2008_p1a,translated_imo_2008_p1b,translated_imo_2008_p6,translated_imo_2009_p2,translated_imo_2010_p2,translated_imo_2011_p6,translated_imo_2012_p5,translated_imo_2014_p4,translated_imo_2015_p3,translated_imo_2018_p1,translated_imo_2019_p2,translated_imo_2019_p6,translated_imo_2020_p1,translated_imo_2021_p3}

UNSOLVED_TAG=${UNSOLVED_TAG:-unsolved_high_budget_v1}
OUT_DIR=outputs/final_eval_imo_ag30_qwen_${UNSOLVED_TAG}
SUMMARY_OUT=outputs/final_eval_imo_ag30_qwen_${UNSOLVED_TAG}_summary.json
mkdir -p "$OUT_DIR"

BEAM_SIZE=${BEAM_SIZE:-64}
SEARCH_DEPTH=${SEARCH_DEPTH:-4}
NUM_RETURN_SEQUENCES=${NUM_RETURN_SEQUENCES:-32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_P=${TOP_P:-0.95}
ROOT_MAX_LEVEL=${ROOT_MAX_LEVEL:-1000}
ROOT_DDAR_TIMEOUT=${ROOT_DDAR_TIMEOUT:-600}
CANDIDATE_MAX_LEVEL=${CANDIDATE_MAX_LEVEL:-300}
CANDIDATE_DDAR_TIMEOUT=${CANDIDATE_DDAR_TIMEOUT:-180}
QWEN_CANDIDATE_WALL_TIMEOUT=${QWEN_CANDIDATE_WALL_TIMEOUT:-120}
QWEN_CANDIDATE_EVAL_LIMIT=${QWEN_CANDIDATE_EVAL_LIMIT:-0}
QWEN_CANDIDATE_DEPTH_EVAL_LIMIT=${QWEN_CANDIDATE_DEPTH_EVAL_LIMIT:-24}
QWEN_CANDIDATE_DDAR_WORKERS=${QWEN_CANDIDATE_DDAR_WORKERS:-8}
QWEN_CANDIDATE_QUALITY_MULTIPLIER=${QWEN_CANDIDATE_QUALITY_MULTIPLIER:-2}
QWEN_CANDIDATE_DSL_FILTER=${QWEN_CANDIDATE_DSL_FILTER:-1}
QWEN_CANDIDATE_DSL_TOKEN_MASK=${QWEN_CANDIDATE_DSL_TOKEN_MASK:-1}
QWEN_CANDIDATE_POINT_REPAIR=${QWEN_CANDIDATE_POINT_REPAIR:-1}
QWEN_CANDIDATE_PROMPT_SAMPLING=${QWEN_CANDIDATE_PROMPT_SAMPLING:-mixed_constructive}
QWEN_CANDIDATE_TEMPLATE_BACKFILL=${QWEN_CANDIDATE_TEMPLATE_BACKFILL:-1}
QWEN_CANDIDATE_RERANK=${QWEN_CANDIDATE_RERANK:-heuristic_diverse}
QWEN_CANDIDATE_VALUE_MODEL=${QWEN_CANDIDATE_VALUE_MODEL:-}
QWEN_LM_FACT_CONTEXT_TOP_K=${QWEN_LM_FACT_CONTEXT_TOP_K:-0}
DRY_RUN=${DRY_RUN:-0}

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

export FINAL_ADAPTER
export BEAM_SIZE SEARCH_DEPTH NUM_RETURN_SEQUENCES MAX_NEW_TOKENS TEMPERATURE TOP_P
export ROOT_MAX_LEVEL ROOT_DDAR_TIMEOUT CANDIDATE_MAX_LEVEL CANDIDATE_DDAR_TIMEOUT
export QWEN_CANDIDATE_WALL_TIMEOUT QWEN_CANDIDATE_EVAL_LIMIT QWEN_CANDIDATE_DEPTH_EVAL_LIMIT QWEN_CANDIDATE_DDAR_WORKERS QWEN_CANDIDATE_QUALITY_MULTIPLIER QWEN_CANDIDATE_DSL_FILTER QWEN_CANDIDATE_DSL_TOKEN_MASK QWEN_CANDIDATE_POINT_REPAIR QWEN_CANDIDATE_PROMPT_SAMPLING QWEN_CANDIDATE_TEMPLATE_BACKFILL QWEN_CANDIDATE_RERANK QWEN_CANDIDATE_VALUE_MODEL QWEN_LM_FACT_CONTEXT_TOP_K

VALUE_MODEL_ARGS=()
if [ -n "$QWEN_CANDIDATE_VALUE_MODEL" ]; then
  VALUE_MODEL_ARGS=(--candidate_value_model "$QWEN_CANDIDATE_VALUE_MODEL")
fi

if [ "$DRY_RUN" = "1" ]; then
  python - "$OUT_DIR" "$UNSOLVED_NAMES" <<'PY'
import json
import os
import sys

summary = {
    'out_dir': sys.argv[1],
    'unsolved_names': [name for name in sys.argv[2].split(',') if name],
    'adapter_path': os.environ.get('FINAL_ADAPTER'),
    'beam_size': int(os.environ.get('BEAM_SIZE', '64')),
    'search_depth': int(os.environ.get('SEARCH_DEPTH', '4')),
    'num_return_sequences': int(os.environ.get('NUM_RETURN_SEQUENCES', '32')),
    'candidate_quality_multiplier': int(os.environ.get('QWEN_CANDIDATE_QUALITY_MULTIPLIER', '2')),
    'candidate_dsl_filter': os.environ.get('QWEN_CANDIDATE_DSL_FILTER', '1') == '1',
    'candidate_dsl_token_mask': os.environ.get('QWEN_CANDIDATE_DSL_TOKEN_MASK', '1') == '1',
    'candidate_point_repair': os.environ.get('QWEN_CANDIDATE_POINT_REPAIR', '1') == '1',
    'candidate_prompt_sampling': os.environ.get('QWEN_CANDIDATE_PROMPT_SAMPLING', 'mixed_constructive'),
    'candidate_template_backfill': os.environ.get('QWEN_CANDIDATE_TEMPLATE_BACKFILL', '1') == '1',
    'candidate_rerank': os.environ.get('QWEN_CANDIDATE_RERANK', 'heuristic_diverse'),
    'candidate_value_model': os.environ.get('QWEN_CANDIDATE_VALUE_MODEL') or None,
    'candidate_wall_timeout': int(os.environ.get('QWEN_CANDIDATE_WALL_TIMEOUT', '120')),
    'candidate_eval_limit': int(os.environ.get('QWEN_CANDIDATE_EVAL_LIMIT', '0')),
    'candidate_depth_eval_limit': int(os.environ.get('QWEN_CANDIDATE_DEPTH_EVAL_LIMIT', '24')),
    'candidate_ddar_workers': int(os.environ.get('QWEN_CANDIDATE_DDAR_WORKERS', '8')),
    'lm_fact_context_top_k': int(os.environ.get('QWEN_LM_FACT_CONTEXT_TOP_K', '0')),
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
  exit 0
fi

xvfb-run -a -s "-screen 0 1024x768x24" python -u scripts/run_qwen_ag_benchmark.py \
  --script_dir scripts \
  --ag_repo repos/alphageometry \
  --problems_file repos/alphageometry/imo_ag_30.txt \
  --defs_file repos/alphageometry/defs.txt \
  --rules_file repos/alphageometry/rules.txt \
  --out_dir "$OUT_DIR" \
  --mode qwen \
  --problem_names "$UNSOLVED_NAMES" \
  --qwen_model models/Qwen2.5-7B \
  --adapter_path "$FINAL_ADAPTER" \
  --dtype bf16 \
  --device_map cuda:0 \
  --root_max_level "$ROOT_MAX_LEVEL" \
  --root_ddar_timeout "$ROOT_DDAR_TIMEOUT" \
  --candidate_max_level "$CANDIDATE_MAX_LEVEL" \
  --candidate_ddar_timeout "$CANDIDATE_DDAR_TIMEOUT" \
  --candidate_wall_timeout "$QWEN_CANDIDATE_WALL_TIMEOUT" \
  --candidate_eval_limit "$QWEN_CANDIDATE_EVAL_LIMIT" \
  --candidate_depth_eval_limit "$QWEN_CANDIDATE_DEPTH_EVAL_LIMIT" \
  --beam_size "$BEAM_SIZE" \
  --search_depth "$SEARCH_DEPTH" \
  --num_return_sequences "$NUM_RETURN_SEQUENCES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
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
  > "$OUT_DIR/run.log" 2>&1

python - "$OUT_DIR" "$SUMMARY_OUT" <<'PY'
import json
import os
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
summary_out = Path(sys.argv[2])
rows = []
for line in (out_dir / 'summary.jsonl').read_text(encoding='utf-8', errors='replace').splitlines():
    if line.strip():
        rows.append(json.loads(line))
solved = [row.get('name') or row.get('problem') for row in rows if row.get('solved')]
aux_solved = [
    {
        'name': row.get('name') or row.get('problem'),
        'depth': row.get('solved_depth'),
        'aux': row.get('aux'),
    }
    for row in rows
    if row.get('solved') and not row.get('root_solved')
]
summary = {
    'out_dir': str(out_dir),
    'num_problems': len(rows),
    'adapter_path': os.environ.get('FINAL_ADAPTER'),
    'solved': len(solved),
    'solved_names': solved,
    'aux_solved': aux_solved,
    'beam_size': int(os.environ.get('BEAM_SIZE', '64')),
    'search_depth': int(os.environ.get('SEARCH_DEPTH', '4')),
    'num_return_sequences': int(os.environ.get('NUM_RETURN_SEQUENCES', '32')),
    'candidate_quality_multiplier': int(os.environ.get('QWEN_CANDIDATE_QUALITY_MULTIPLIER', '2')),
    'candidate_dsl_filter': os.environ.get('QWEN_CANDIDATE_DSL_FILTER', '1') == '1',
    'candidate_dsl_token_mask': os.environ.get('QWEN_CANDIDATE_DSL_TOKEN_MASK', '1') == '1',
    'candidate_point_repair': os.environ.get('QWEN_CANDIDATE_POINT_REPAIR', '1') == '1',
    'candidate_prompt_sampling': os.environ.get('QWEN_CANDIDATE_PROMPT_SAMPLING', 'mixed_constructive'),
    'candidate_template_backfill': os.environ.get('QWEN_CANDIDATE_TEMPLATE_BACKFILL', '1') == '1',
    'candidate_rerank': os.environ.get('QWEN_CANDIDATE_RERANK', 'heuristic_diverse'),
    'candidate_value_model': os.environ.get('QWEN_CANDIDATE_VALUE_MODEL') or None,
    'candidate_wall_timeout': int(os.environ.get('QWEN_CANDIDATE_WALL_TIMEOUT', '120')),
    'candidate_eval_limit': int(os.environ.get('QWEN_CANDIDATE_EVAL_LIMIT', '0')),
    'candidate_depth_eval_limit': int(os.environ.get('QWEN_CANDIDATE_DEPTH_EVAL_LIMIT', '24')),
    'candidate_ddar_workers': int(os.environ.get('QWEN_CANDIDATE_DDAR_WORKERS', '8')),
    'lm_fact_context_top_k': int(os.environ.get('QWEN_LM_FACT_CONTEXT_TOP_K', '0')),
}
summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
