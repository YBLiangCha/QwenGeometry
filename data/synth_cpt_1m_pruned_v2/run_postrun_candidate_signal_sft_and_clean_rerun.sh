#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:-/root/rivermind-data/qwen_ag_lm}
. "$WORK/env.sh" 2>/dev/null || true
. "$WORK/venv/bin/activate" 2>/dev/null || true
cd "$WORK"

SCRIPT_DIR=${SCRIPT_DIR:-scripts}
OLD_TAG=${OLD_TAG:-unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1}
OLD_OUT_DIR=${OLD_OUT_DIR:-outputs/final_eval_imo_ag30_qwen_${OLD_TAG}}
OLD_EVENTS_DIR=${OLD_EVENTS_DIR:-$OLD_OUT_DIR/events}
OLD_SUMMARY_JSONL=${OLD_SUMMARY_JSONL:-$OLD_OUT_DIR/summary.jsonl}
BASELINE_SUMMARY_JSONL=${BASELINE_SUMMARY_JSONL:-outputs/final_eval_imo_ag30_qwen_unsolved_high_budget_value_v3_cost_template_depth16_v1/summary.jsonl}

WAIT_FOR_OLD_BENCH=${WAIT_FOR_OLD_BENCH:-1}
WAIT_INTERVAL=${WAIT_INTERVAL:-300}
POSTRUN_TAG=${POSTRUN_TAG:-postrun_v1}
STAGED=${STAGED:-data/staged_1m_pruned_v2}

FACT_TAG=${FACT_TAG:-factctx_promptaug_top8_stage2max2000_v1}
FACT_DIR=${FACT_DIR:-$STAGED/$FACT_TAG}
FACT_MIX_TRAIN=${FACT_MIX_TRAIN:-$FACT_DIR/fact_context_mixed_train.jsonl}
FACT_MIX_EVAL=${FACT_MIX_EVAL:-$FACT_DIR/fact_context_mixed_eval.jsonl}
BASE_ADAPTER=${BASE_ADAPTER:-outputs/stage2_fact_context_after_stage2_lora_qwen2_5_7b_${FACT_TAG}}

DATA_TAG=${DATA_TAG:-candidate_signals_${OLD_TAG}_${POSTRUN_TAG}}
DATA_DIR=${DATA_DIR:-$STAGED/$DATA_TAG}
SIGNAL_TRAIN=${SIGNAL_TRAIN:-$DATA_DIR/candidate_signal_aux_train.jsonl}
SIGNAL_EVAL=${SIGNAL_EVAL:-$DATA_DIR/candidate_signal_aux_eval.jsonl}
SIGNAL_SUMMARY=${SIGNAL_SUMMARY:-$DATA_DIR/summary.json}
HARD_NEG_TRAIN=${HARD_NEG_TRAIN:-$DATA_DIR/candidate_hard_negative_aux_train.jsonl}
HARD_NEG_EVAL=${HARD_NEG_EVAL:-$DATA_DIR/candidate_hard_negative_aux_eval.jsonl}
HARD_NEG_SUMMARY=${HARD_NEG_SUMMARY:-$DATA_DIR/hard_negative_summary.json}

SFT_TAG=${SFT_TAG:-candidate_signal_sft_${OLD_TAG}_${POSTRUN_TAG}}
SFT_WORK_DIR=${SFT_WORK_DIR:-$STAGED/$SFT_TAG}
MIXED_TRAIN=${MIXED_TRAIN:-$SFT_WORK_DIR/candidate_signal_mixed_train.jsonl}
MIXED_EVAL=${MIXED_EVAL:-$SFT_WORK_DIR/candidate_signal_mixed_eval.jsonl}
SFT_OUT=${SFT_OUT:-outputs/stage3_candidate_signal_after_factctx_lora_qwen2_5_7b_${SFT_TAG}}
RUN_SUMMARY=${RUN_SUMMARY:-$SFT_WORK_DIR/summary.json}

MAX_FACT_MIX_ROWS=${MAX_FACT_MIX_ROWS:-2000}
MAX_FACT_EVAL_ROWS=${MAX_FACT_EVAL_ROWS:-400}
SIGNAL_REPEAT=${SIGNAL_REPEAT:-4}
SIGNAL_MIN_PROGRESS_DELTA=${SIGNAL_MIN_PROGRESS_DELTA:-40}
SIGNAL_MAX_ELAPSED_SEC=${SIGNAL_MAX_ELAPSED_SEC:-120}
SIGNAL_MIN_PROGRESS_EFFICIENCY=${SIGNAL_MIN_PROGRESS_EFFICIENCY:-0.5}
SIGNAL_MAX_PROGRESS_ROWS_PER_PROBLEM=${SIGNAL_MAX_PROGRESS_ROWS_PER_PROBLEM:-16}
SIGNAL_MAX_PROGRESS_ROWS_PER_TYPE=${SIGNAL_MAX_PROGRESS_ROWS_PER_TYPE:-32}
SIGNAL_SOLVED_REPEAT=${SIGNAL_SOLVED_REPEAT:-16}
USE_HARD_NEGATIVES=${USE_HARD_NEGATIVES:-1}
UNLIKELIHOOD_WEIGHT=${UNLIKELIHOOD_WEIGHT:-0.1}
TRAIN_SFT=${TRAIN_SFT:-1}
FORCE_TRAIN=${FORCE_TRAIN:-0}

RUN_CLEAN_RERUN=${RUN_CLEAN_RERUN:-0}
CLEAN_CANDIDATE_EVAL_LIMIT=${CLEAN_CANDIDATE_EVAL_LIMIT:-0}
CLEAN_CANDIDATE_DEPTH_EVAL_LIMIT=${CLEAN_CANDIDATE_DEPTH_EVAL_LIMIT:-48}
CLEAN_CANDIDATE_TEMPLATE_BACKFILL_EXTRA_SLOTS=${CLEAN_CANDIDATE_TEMPLATE_BACKFILL_EXTRA_SLOTS:-0}
CLEAN_CANDIDATE_DEPTH_TYPE_EVAL_CAP=${CLEAN_CANDIDATE_DEPTH_TYPE_EVAL_CAP:-0}
CLEAN_CANDIDATE_DEPTH_TEMPLATE_EVAL_SLOTS=${CLEAN_CANDIDATE_DEPTH_TEMPLATE_EVAL_SLOTS:-0}
CLEAN_CANDIDATE_ADAPTIVE_TYPE_PENALTY=${CLEAN_CANDIDATE_ADAPTIVE_TYPE_PENALTY:-0}
CLEAN_CANDIDATE_ADAPTIVE_TYPE_PENALTY_THRESHOLD=${CLEAN_CANDIDATE_ADAPTIVE_TYPE_PENALTY_THRESHOLD:-32}
CLEAN_CANDIDATE_ADAPTIVE_TYPE_PENALTY_WEIGHT=${CLEAN_CANDIDATE_ADAPTIVE_TYPE_PENALTY_WEIGHT:-0.35}
CLEAN_CANDIDATE_ADAPTIVE_TYPE_PENALTY_MAX=${CLEAN_CANDIDATE_ADAPTIVE_TYPE_PENALTY_MAX:-1.5}
CLEAN_CANDIDATE_ADAPTIVE_TYPE_PENALTY_REASONS=${CLEAN_CANDIDATE_ADAPTIVE_TYPE_PENALTY_REASONS:-point_too_close,point_too_far,point_already_exists,unknown_point,invalid_quad_solve,dep_check_fail,invalid_line_intersect,value_error,invalid_predicate}
CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_ANCHOR=${CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_ANCHOR:-0}
CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_DELTA=${CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_DELTA:-60}
CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_DELTA_FLOOR=${CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_DELTA_FLOOR:-25}
CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_ROOT_RATIO=${CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_ROOT_RATIO:-0.6}
CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_BASE=${CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_BASE:-2.0}
CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_WEIGHT=${CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_WEIGHT:-0.08}
CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_MAX=${CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_MAX:-3.2}
CLEAN_CANDIDATE_STATIC_PROGRESS_TYPE_BONUS=${CLEAN_CANDIDATE_STATIC_PROGRESS_TYPE_BONUS:-${CANDIDATE_STATIC_PROGRESS_TYPE_BONUS:-}}
CLEAN_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS=${CLEAN_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS:-0}
CLEAN_CANDIDATE_DEPTH_TAIL_EVAL_STRATEGY=${CLEAN_CANDIDATE_DEPTH_TAIL_EVAL_STRATEGY:-even}
CLEAN_TIMEOUT_BEAM_FALLBACK_LIMIT=${CLEAN_TIMEOUT_BEAM_FALLBACK_LIMIT:-0}
CLEAN_TIMEOUT_BEAM_FALLBACK_MODE=${CLEAN_TIMEOUT_BEAM_FALLBACK_MODE:-empty}
CLEAN_CANDIDATE_DDAR_TIMEOUT=${CLEAN_CANDIDATE_DDAR_TIMEOUT:-240}
CLEAN_CANDIDATE_WALL_TIMEOUT=${CLEAN_CANDIDATE_WALL_TIMEOUT:-150}
CLEAN_CANDIDATE_SOFT_TIMEOUT_MARGIN_SEC=${CLEAN_CANDIDATE_SOFT_TIMEOUT_MARGIN_SEC:-5}
CLEAN_CANDIDATE_DDAR_WORKERS=${CLEAN_CANDIDATE_DDAR_WORKERS:-8}
CLEAN_CANDIDATE_BEAM_SCORE=${CLEAN_CANDIDATE_BEAM_SCORE:-lm_score}
CLEAN_CANDIDATE_BEAM_PROGRESS_WEIGHT=${CLEAN_CANDIDATE_BEAM_PROGRESS_WEIGHT:-0.0}
CLEAN_CANDIDATE_BEAM_PROGRESS_CAP=${CLEAN_CANDIDATE_BEAM_PROGRESS_CAP:-4.0}
CLEAN_CANDIDATE_DECODE_BEAM_LIMIT=${CLEAN_CANDIDATE_DECODE_BEAM_LIMIT:-0}
CLEAN_CANDIDATE_PROMPT_SAMPLING=${CLEAN_CANDIDATE_PROMPT_SAMPLING:-mixed_constructive}
CLEAN_CANDIDATE_PROMPT_PREFERRED_TYPE_LIMIT=${CLEAN_CANDIDATE_PROMPT_PREFERRED_TYPE_LIMIT:-12}
CLEAN_CANDIDATE_TEMPLATE_PREFERRED_TYPE_LIMIT=${CLEAN_CANDIDATE_TEMPLATE_PREFERRED_TYPE_LIMIT:-36}
CLEAN_LM_FACT_CONTEXT_TOP_K=${CLEAN_LM_FACT_CONTEXT_TOP_K:-8}
CLEAN_BEAM_SIZE=${CLEAN_BEAM_SIZE:-64}
CLEAN_SEARCH_DEPTH=${CLEAN_SEARCH_DEPTH:-4}
CLEAN_NUM_RETURN_SEQUENCES=${CLEAN_NUM_RETURN_SEQUENCES:-48}
CLEAN_CANDIDATE_QUALITY_MULTIPLIER=${CLEAN_CANDIDATE_QUALITY_MULTIPLIER:-3}
CLEAN_PROBLEM_NAMES=${CLEAN_PROBLEM_NAMES:-translated_imo_2000_p6,translated_imo_2004_p1,translated_imo_2008_p1a,translated_imo_2008_p1b,translated_imo_2008_p6,translated_imo_2009_p2,translated_imo_2010_p2,translated_imo_2011_p6,translated_imo_2012_p5,translated_imo_2014_p4,translated_imo_2015_p3,translated_imo_2018_p1,translated_imo_2019_p2,translated_imo_2019_p6,translated_imo_2020_p1,translated_imo_2021_p3}
CLEAN_CANDIDATE_RERANK=${CLEAN_CANDIDATE_RERANK:-value_model_diverse}
CLEAN_SECONDARY_VALUE_MODEL=${CLEAN_SECONDARY_VALUE_MODEL:-}
CLEAN_FRONTFILL_LIMIT=${CLEAN_FRONTFILL_LIMIT:-8}
CLEAN_RERUN_TAG=${CLEAN_RERUN_TAG:-unsolved_factctx_promptaug_top8_candidate_signal_${POSTRUN_TAG}_value_v12_grammar_semantic_v4_scores_dedup_dupneg_depth${CLEAN_CANDIDATE_DEPTH_EVAL_LIMIT}_nrs${CLEAN_NUM_RETURN_SEQUENCES}_qm${CLEAN_CANDIDATE_QUALITY_MULTIPLIER}_v1}
CLEAN_OUT_DIR=${CLEAN_OUT_DIR:-outputs/final_eval_imo_ag30_qwen_${CLEAN_RERUN_TAG}}
VALUE_MODEL=${VALUE_MODEL:-outputs/candidate_value_model_v12_logistic_preddar_nodup_semantic_v3_partial7events6summary_v1/candidate_value_model.json}

LOG=${LOG:-outputs/${SFT_TAG}.postrun_queue.log}
mkdir -p "$(dirname "$LOG")" "$DATA_DIR" "$SFT_WORK_DIR"
export ANALYSIS_JSON REPORT_MD HARD_NEG_TRAIN HARD_NEG_EVAL HARD_NEG_SUMMARY
export USE_HARD_NEGATIVES UNLIKELIHOOD_WEIGHT MAX_FACT_MIX_ROWS MAX_FACT_EVAL_ROWS SIGNAL_REPEAT
export SIGNAL_MIN_PROGRESS_DELTA SIGNAL_MAX_ELAPSED_SEC SIGNAL_MIN_PROGRESS_EFFICIENCY
export SIGNAL_MAX_PROGRESS_ROWS_PER_PROBLEM SIGNAL_MAX_PROGRESS_ROWS_PER_TYPE SIGNAL_SOLVED_REPEAT

log() {
  date '+%F %T %z' | tr -d '\n' | tee -a "$LOG"
  printf ' %s\n' "$*" | tee -a "$LOG"
}

if [ "$WAIT_FOR_OLD_BENCH" = "1" ]; then
  log "waiting for old benchmark to exit: $OLD_TAG"
  while pgrep -f "run_qwen_ag_benchmark.py.*${OLD_TAG}" >/dev/null; do
    pgrep -af "run_qwen_ag_benchmark.py.*${OLD_TAG}" | tee -a "$LOG" || true
    sleep "$WAIT_INTERVAL"
  done
fi

log "old benchmark no longer active; building postrun artifacts"
if [ ! -d "$OLD_EVENTS_DIR" ]; then
  echo "missing events dir: $OLD_EVENTS_DIR" | tee -a "$LOG" >&2
  exit 1
fi

ANALYSIS_JSON=${ANALYSIS_JSON:-outputs/${OLD_TAG}_${POSTRUN_TAG}_analysis.json}
REPORT_MD=${REPORT_MD:-outputs/${OLD_TAG}_${POSTRUN_TAG}_report.md}
python -u "$SCRIPT_DIR/analyze_qwen_ag_events.py" \
  --out_dir "$OLD_OUT_DIR" \
  --baseline_summary_jsonl "$BASELINE_SUMMARY_JSONL" \
  --out_file "$ANALYSIS_JSON" \
  --top_problems 16 \
  >> "$LOG" 2>&1
python -u "$SCRIPT_DIR/report_qwen_ag_analysis.py" \
  --analysis_json "$ANALYSIS_JSON" \
  --out_file "$REPORT_MD" \
  >> "$LOG" 2>&1

python -u "$SCRIPT_DIR/build_aux_sft_from_candidate_signals.py" \
  --events_dir "$OLD_EVENTS_DIR" \
  --train_file "$SIGNAL_TRAIN" \
  --eval_file "$SIGNAL_EVAL" \
  --summary_file "$SIGNAL_SUMMARY" \
  --min_progress_delta "$SIGNAL_MIN_PROGRESS_DELTA" \
  --max_elapsed_sec "$SIGNAL_MAX_ELAPSED_SEC" \
  --min_progress_efficiency "$SIGNAL_MIN_PROGRESS_EFFICIENCY" \
  --max_progress_rows_per_problem "$SIGNAL_MAX_PROGRESS_ROWS_PER_PROBLEM" \
  --max_progress_rows_per_type "$SIGNAL_MAX_PROGRESS_ROWS_PER_TYPE" \
  --solved_repeat "$SIGNAL_SOLVED_REPEAT" \
  >> "$LOG" 2>&1
python -u "$SCRIPT_DIR/build_aux_hard_negative_from_candidate_signals.py" \
  --events_dir "$OLD_EVENTS_DIR" \
  --train_file "$HARD_NEG_TRAIN" \
  --eval_file "$HARD_NEG_EVAL" \
  --summary_file "$HARD_NEG_SUMMARY" \
  >> "$LOG" 2>&1

python - "$SIGNAL_TRAIN" "$SIGNAL_EVAL" "$FACT_MIX_TRAIN" "$FACT_MIX_EVAL" "$MIXED_TRAIN" "$MIXED_EVAL" "$RUN_SUMMARY" <<'PY'
import json
import os
import sys
from pathlib import Path

signal_train, signal_eval, fact_train, fact_eval, mixed_train, mixed_eval, run_summary = map(Path, sys.argv[1:])
max_fact_train = int(os.environ.get('MAX_FACT_MIX_ROWS', '2000'))
max_fact_eval = int(os.environ.get('MAX_FACT_EVAL_ROWS', '400'))
signal_repeat = max(1, int(os.environ.get('SIGNAL_REPEAT', '4')))

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
mixed_train_rows = signal_train_rows * signal_repeat + fact_train_rows
mixed_eval_rows = signal_eval_rows + fact_eval_rows
write_jsonl(mixed_train, mixed_train_rows)
write_jsonl(mixed_eval, mixed_eval_rows)

summary = {
    'status': 'prepared',
    'signal_train_rows': len(signal_train_rows),
    'signal_eval_rows': len(signal_eval_rows),
    'signal_repeat': signal_repeat,
    'fact_train_rows': len(fact_train_rows),
    'fact_eval_rows': len(fact_eval_rows),
    'mixed_train_rows': len(mixed_train_rows),
    'mixed_eval_rows': len(mixed_eval_rows),
    'hard_negative_train': os.environ.get('HARD_NEG_TRAIN'),
    'hard_negative_eval': os.environ.get('HARD_NEG_EVAL'),
    'hard_negative_summary': os.environ.get('HARD_NEG_SUMMARY'),
    'use_hard_negatives': os.environ.get('USE_HARD_NEGATIVES') == '1',
    'unlikelihood_weight': float(os.environ.get('UNLIKELIHOOD_WEIGHT', '0.1')),
    'analysis_json': os.environ.get('ANALYSIS_JSON'),
    'report_md': os.environ.get('REPORT_MD'),
    'mixed_train': str(mixed_train),
    'mixed_eval': str(mixed_eval),
}
run_summary.parent.mkdir(parents=True, exist_ok=True)
run_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

if [ "$TRAIN_SFT" = "1" ]; then
  if [ -s "$SFT_OUT/adapter_model.safetensors" ]; then
    log "SFT adapter already exists; skipping train: $SFT_OUT"
  else
    if [ -d "$SFT_OUT" ] && [ "$FORCE_TRAIN" != "1" ]; then
      echo "SFT output exists without adapter; set FORCE_TRAIN=1 to overwrite: $SFT_OUT" | tee -a "$LOG" >&2
      exit 1
    fi
    if [ "$FORCE_TRAIN" = "1" ]; then
      rm -rf "$SFT_OUT"
    fi
    mkdir -p "$SFT_OUT"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
    NEG_ARGS=()
    if [ "$USE_HARD_NEGATIVES" = "1" ]; then
      NEG_ARGS=(
        --negative_train_file "$HARD_NEG_TRAIN"
        --negative_eval_file "$HARD_NEG_EVAL"
        --unlikelihood_weight "$UNLIKELIHOOD_WEIGHT"
      )
    fi
    log "training candidate-signal adapter: $SFT_OUT"
    python -u "$SCRIPT_DIR/train_qwen_aux_lora.py" \
      --model_name_or_path models/Qwen2.5-7B \
      --init_adapter_path "$BASE_ADAPTER" \
      --train_file "$MIXED_TRAIN" \
      --eval_file "$MIXED_EVAL" \
      --output_dir "$SFT_OUT" \
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
      >> "$SFT_OUT/train.log" 2>&1
  fi
fi

python - "$RUN_SUMMARY" "$SFT_OUT" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
summary = json.loads(summary_path.read_text(encoding='utf-8'))
summary['status'] = 'trained' if Path(sys.argv[2], 'adapter_model.safetensors').exists() else summary.get('status')
summary['output_dir'] = sys.argv[2]
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

if [ "$RUN_CLEAN_RERUN" = "1" ]; then
  if [ ! -s "$SFT_OUT/adapter_model.safetensors" ]; then
    echo "missing trained adapter for clean rerun: $SFT_OUT" | tee -a "$LOG" >&2
    exit 1
  fi
  if [ -e "$CLEAN_OUT_DIR/summary.jsonl" ]; then
    echo "clean rerun output already exists: $CLEAN_OUT_DIR" | tee -a "$LOG" >&2
    exit 1
  fi
  CLEAN_SECONDARY_VALUE_MODEL_ARGS=()
  if [ -n "$CLEAN_SECONDARY_VALUE_MODEL" ]; then
    if [ ! -s "$CLEAN_SECONDARY_VALUE_MODEL" ]; then
      echo "missing CLEAN_SECONDARY_VALUE_MODEL: $CLEAN_SECONDARY_VALUE_MODEL" | tee -a "$LOG" >&2
      exit 1
    fi
    CLEAN_SECONDARY_VALUE_MODEL_ARGS=(
      --candidate_secondary_value_model "$CLEAN_SECONDARY_VALUE_MODEL"
      --candidate_frontfill_limit "$CLEAN_FRONTFILL_LIMIT"
    )
  fi
  CLEAN_ADAPTIVE_TYPE_PENALTY_ARGS=()
  if [ "$CLEAN_CANDIDATE_ADAPTIVE_TYPE_PENALTY" = "1" ]; then
    CLEAN_ADAPTIVE_TYPE_PENALTY_ARGS=(
      --candidate_adaptive_type_penalty
      --candidate_adaptive_type_penalty_threshold "$CLEAN_CANDIDATE_ADAPTIVE_TYPE_PENALTY_THRESHOLD"
      --candidate_adaptive_type_penalty_weight "$CLEAN_CANDIDATE_ADAPTIVE_TYPE_PENALTY_WEIGHT"
      --candidate_adaptive_type_penalty_max "$CLEAN_CANDIDATE_ADAPTIVE_TYPE_PENALTY_MAX"
      --candidate_adaptive_type_penalty_reasons "$CLEAN_CANDIDATE_ADAPTIVE_TYPE_PENALTY_REASONS"
    )
  fi
  CLEAN_DYNAMIC_PROGRESS_TYPE_ANCHOR_ARGS=()
  if [ "$CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_ANCHOR" = "1" ]; then
    CLEAN_DYNAMIC_PROGRESS_TYPE_ANCHOR_ARGS=(
      --candidate_dynamic_progress_type_anchor
      --candidate_dynamic_progress_type_min_delta "$CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_DELTA"
      --candidate_dynamic_progress_type_min_delta_floor "$CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_DELTA_FLOOR"
      --candidate_dynamic_progress_type_min_root_ratio "$CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_ROOT_RATIO"
      --candidate_dynamic_progress_type_bonus_base "$CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_BASE"
      --candidate_dynamic_progress_type_bonus_weight "$CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_WEIGHT"
      --candidate_dynamic_progress_type_bonus_max "$CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_MAX"
    )
  fi
  CLEAN_STATIC_PROGRESS_TYPE_BONUS_ARGS=()
  if [ -n "$CLEAN_CANDIDATE_STATIC_PROGRESS_TYPE_BONUS" ]; then
    if [ ! -s "$CLEAN_CANDIDATE_STATIC_PROGRESS_TYPE_BONUS" ]; then
      echo "missing CLEAN_CANDIDATE_STATIC_PROGRESS_TYPE_BONUS: $CLEAN_CANDIDATE_STATIC_PROGRESS_TYPE_BONUS" | tee -a "$LOG" >&2
      exit 1
    fi
    CLEAN_STATIC_PROGRESS_TYPE_BONUS_ARGS=(
      --candidate_static_progress_type_bonus "$CLEAN_CANDIDATE_STATIC_PROGRESS_TYPE_BONUS"
    )
  fi
  log "starting clean rerun: $CLEAN_RERUN_TAG"
  log "clean rerun candidate eval limit: ${CLEAN_CANDIDATE_EVAL_LIMIT}; depth eval limit: ${CLEAN_CANDIDATE_DEPTH_EVAL_LIMIT}; template backfill extra slots: ${CLEAN_CANDIDATE_TEMPLATE_BACKFILL_EXTRA_SLOTS}; depth type cap: ${CLEAN_CANDIDATE_DEPTH_TYPE_EVAL_CAP}; depth template slots: ${CLEAN_CANDIDATE_DEPTH_TEMPLATE_EVAL_SLOTS}; adaptive type penalty: ${CLEAN_CANDIDATE_ADAPTIVE_TYPE_PENALTY}/thr${CLEAN_CANDIDATE_ADAPTIVE_TYPE_PENALTY_THRESHOLD}/w${CLEAN_CANDIDATE_ADAPTIVE_TYPE_PENALTY_WEIGHT}/max${CLEAN_CANDIDATE_ADAPTIVE_TYPE_PENALTY_MAX}; dynamic progress type anchor: ${CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_ANCHOR}/min${CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_DELTA}/floor${CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_DELTA_FLOOR}/ratio${CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_ROOT_RATIO}/base${CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_BASE}/w${CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_WEIGHT}/max${CLEAN_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_MAX}; static progress type bonus: ${CLEAN_CANDIDATE_STATIC_PROGRESS_TYPE_BONUS:-none}; depth tail slots: ${CLEAN_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS}; depth tail strategy: ${CLEAN_CANDIDATE_DEPTH_TAIL_EVAL_STRATEGY}; decode beam limit: ${CLEAN_CANDIDATE_DECODE_BEAM_LIMIT}; timeout beam fallback: ${CLEAN_TIMEOUT_BEAM_FALLBACK_LIMIT}; timeout fallback mode: ${CLEAN_TIMEOUT_BEAM_FALLBACK_MODE}; candidate timeout: ${CLEAN_CANDIDATE_DDAR_TIMEOUT}; wall timeout: ${CLEAN_CANDIDATE_WALL_TIMEOUT}; soft timeout margin: ${CLEAN_CANDIDATE_SOFT_TIMEOUT_MARGIN_SEC}; workers: ${CLEAN_CANDIDATE_DDAR_WORKERS}; beam score: ${CLEAN_CANDIDATE_BEAM_SCORE}; progress weight: ${CLEAN_CANDIDATE_BEAM_PROGRESS_WEIGHT}; progress cap: ${CLEAN_CANDIDATE_BEAM_PROGRESS_CAP}; prompt sampling: ${CLEAN_CANDIDATE_PROMPT_SAMPLING}; prompt_preferred_type_limit: ${CLEAN_CANDIDATE_PROMPT_PREFERRED_TYPE_LIMIT}; template_preferred_type_limit: ${CLEAN_CANDIDATE_TEMPLATE_PREFERRED_TYPE_LIMIT}; lm fact top-k: ${CLEAN_LM_FACT_CONTEXT_TOP_K}; beam: ${CLEAN_BEAM_SIZE}; search depth: ${CLEAN_SEARCH_DEPTH}; nrs: ${CLEAN_NUM_RETURN_SEQUENCES}; quality multiplier: ${CLEAN_CANDIDATE_QUALITY_MULTIPLIER}; rerank: ${CLEAN_CANDIDATE_RERANK}; frontfill: ${CLEAN_FRONTFILL_LIMIT}; value model: ${VALUE_MODEL}; secondary value model: ${CLEAN_SECONDARY_VALUE_MODEL:-none}"
  log "clean rerun problem names: ${CLEAN_PROBLEM_NAMES}"
  xvfb-run -a -s "-screen 0 1024x768x24" python -u "$SCRIPT_DIR/run_qwen_ag_benchmark.py" \
    --script_dir "$SCRIPT_DIR" \
    --ag_repo repos/alphageometry \
    --problems_file repos/alphageometry/imo_ag_30.txt \
    --defs_file repos/alphageometry/defs.txt \
    --rules_file repos/alphageometry/rules.txt \
    --out_dir "$CLEAN_OUT_DIR" \
    --mode qwen \
    --problem_names "$CLEAN_PROBLEM_NAMES" \
    --qwen_model models/Qwen2.5-7B \
    --adapter_path "$SFT_OUT" \
    --dtype bf16 \
    --device_map cuda:0 \
    --root_max_level 1000 \
    --root_ddar_timeout 600 \
    --candidate_max_level 300 \
    --candidate_ddar_timeout "$CLEAN_CANDIDATE_DDAR_TIMEOUT" \
    --candidate_wall_timeout "$CLEAN_CANDIDATE_WALL_TIMEOUT" \
    --candidate_soft_timeout_margin_sec "$CLEAN_CANDIDATE_SOFT_TIMEOUT_MARGIN_SEC" \
    --candidate_eval_limit "$CLEAN_CANDIDATE_EVAL_LIMIT" \
    --candidate_depth_eval_limit "$CLEAN_CANDIDATE_DEPTH_EVAL_LIMIT" \
    --candidate_template_backfill_extra_slots "$CLEAN_CANDIDATE_TEMPLATE_BACKFILL_EXTRA_SLOTS" \
    --candidate_depth_type_eval_cap "$CLEAN_CANDIDATE_DEPTH_TYPE_EVAL_CAP" \
    --candidate_depth_template_eval_slots "$CLEAN_CANDIDATE_DEPTH_TEMPLATE_EVAL_SLOTS" \
    "${CLEAN_ADAPTIVE_TYPE_PENALTY_ARGS[@]}" \
    "${CLEAN_DYNAMIC_PROGRESS_TYPE_ANCHOR_ARGS[@]}" \
    "${CLEAN_STATIC_PROGRESS_TYPE_BONUS_ARGS[@]}" \
    --candidate_depth_tail_eval_slots "$CLEAN_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS" \
    --candidate_depth_tail_eval_strategy "$CLEAN_CANDIDATE_DEPTH_TAIL_EVAL_STRATEGY" \
    --candidate_timeout_beam_fallback_limit "$CLEAN_TIMEOUT_BEAM_FALLBACK_LIMIT" \
    --candidate_timeout_beam_fallback_mode "$CLEAN_TIMEOUT_BEAM_FALLBACK_MODE" \
    --beam_size "$CLEAN_BEAM_SIZE" \
    --search_depth "$CLEAN_SEARCH_DEPTH" \
    --num_return_sequences "$CLEAN_NUM_RETURN_SEQUENCES" \
    --max_new_tokens 64 \
    --temperature 0.8 \
    --top_p 0.95 \
    --candidate_quality_multiplier "$CLEAN_CANDIDATE_QUALITY_MULTIPLIER" \
    --candidate_dsl_filter \
    --candidate_dsl_token_mask \
    --candidate_point_repair \
    --candidate_point_mask \
    --candidate_canonical_dedup \
    --candidate_prompt_sampling "$CLEAN_CANDIDATE_PROMPT_SAMPLING" \
    --candidate_prompt_preferred_type_limit "$CLEAN_CANDIDATE_PROMPT_PREFERRED_TYPE_LIMIT" \
    --candidate_template_preferred_type_limit "$CLEAN_CANDIDATE_TEMPLATE_PREFERRED_TYPE_LIMIT" \
    --candidate_template_backfill \
    --candidate_rerank "$CLEAN_CANDIDATE_RERANK" \
    --candidate_value_model "$VALUE_MODEL" \
    "${CLEAN_SECONDARY_VALUE_MODEL_ARGS[@]}" \
    --candidate_beam_score "$CLEAN_CANDIDATE_BEAM_SCORE" \
    --candidate_beam_progress_weight "$CLEAN_CANDIDATE_BEAM_PROGRESS_WEIGHT" \
    --candidate_beam_progress_cap "$CLEAN_CANDIDATE_BEAM_PROGRESS_CAP" \
    --candidate_decode_beam_limit "$CLEAN_CANDIDATE_DECODE_BEAM_LIMIT" \
    --candidate_ddar_workers "$CLEAN_CANDIDATE_DDAR_WORKERS" \
    --lm_fact_context_top_k "$CLEAN_LM_FACT_CONTEXT_TOP_K" \
    >> "outputs/${CLEAN_RERUN_TAG}.log" 2>&1
fi

log "postrun queue finished"
