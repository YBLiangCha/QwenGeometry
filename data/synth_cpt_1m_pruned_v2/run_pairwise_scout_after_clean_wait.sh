#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:-/root/rivermind-data/qwen_ag_lm}
. "$WORK/env.sh" 2>/dev/null || true
. "$WORK/venv/bin/activate" 2>/dev/null || true
cd "$WORK"

SCRIPT_DIR=${SCRIPT_DIR:-scripts}
WAIT_OUT_DIR=${WAIT_OUT_DIR:-outputs/final_eval_imo_ag30_qwen_unsolved_factctx_promptaug_top8_candidate_signal_postrun_value_v12_default_v1_depth48_t240_w150_nrs48_qm3_sigrep4_blinedia_statededup_nodediv_dsltpl_combotpl_rarecombo_vprior_v1}
WAIT_SUMMARY_JSONL=${WAIT_SUMMARY_JSONL:-$WAIT_OUT_DIR/summary.jsonl}
WAIT_PROCESS_PATTERN=${WAIT_PROCESS_PATTERN:-run_qwen_ag_benchmark.py.*$(basename "$WAIT_OUT_DIR")}
WAIT_EXPECTED_ROWS=${WAIT_EXPECTED_ROWS:-16}
WAIT_MIN_ROWS=${WAIT_MIN_ROWS:-1}
WAIT_INTERVAL=${WAIT_INTERVAL:-60}
WAIT_ALLOW_INCOMPLETE=${WAIT_ALLOW_INCOMPLETE:-0}

SCOUT_TIMEOUT_BEAM_FALLBACK_LIMIT=${SCOUT_TIMEOUT_BEAM_FALLBACK_LIMIT:-4}
SCOUT_TIMEOUT_BEAM_FALLBACK_MODE=${SCOUT_TIMEOUT_BEAM_FALLBACK_MODE:-append}
SCOUT_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS=${SCOUT_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS:-4}
SCOUT_CANDIDATE_DEPTH_TAIL_EVAL_STRATEGY=${SCOUT_CANDIDATE_DEPTH_TAIL_EVAL_STRATEGY:-near_spread}
SCOUT_TAG=${SCOUT_TAG:-unsolved_v54eqdistance_pair_bonus_scout_d16_tc4_tail${SCOUT_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS}_fact12_t160_w100_fb${SCOUT_TIMEOUT_BEAM_FALLBACK_LIMIT}append_v1}
SCOUT_OUT_DIR=${SCOUT_OUT_DIR:-outputs/final_eval_imo_ag30_qwen_${SCOUT_TAG}}
SCOUT_LOG=${SCOUT_LOG:-outputs/${SCOUT_TAG}.log}
SCOUT_QUEUE_LOG=${SCOUT_QUEUE_LOG:-outputs/${SCOUT_TAG}.queue.log}
SCOUT_PROBLEM_NAMES=${SCOUT_PROBLEM_NAMES:-}

QWEN_MODEL=${QWEN_MODEL:-models/Qwen2.5-7B}
ADAPTER_PATH=${ADAPTER_PATH:-outputs/stage3_candidate_signal_after_factctx_lora_qwen2_5_7b_candidate_signal_sft_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_postrun_value_v12_default_v1}
VALUE_MODEL=${VALUE_MODEL:-outputs/candidate_value_model_v12_logistic_preddar_nodup_semantic_v3_partial7events6summary_v1/candidate_value_model.json}
SECONDARY_VALUE_MODEL=${SECONDARY_VALUE_MODEL:-outputs/candidate_value_model_v16_pairwise_solved_biased_progress_filter_oldfull_current4_v1/candidate_value_model.json}
SCOUT_RERANK=${SCOUT_RERANK:-value_model_frontfill_progress_diverse}
SCOUT_FRONTFILL_LIMIT=${SCOUT_FRONTFILL_LIMIT:-8}
TRAIN_SCOUT_VALUE_MODEL=${TRAIN_SCOUT_VALUE_MODEL:-1}
SCOUT_REFRESH_VALUE_ROLE=${SCOUT_REFRESH_VALUE_ROLE:-secondary}
VALUE_MODEL_APPEND_SCRIPT=${VALUE_MODEL_APPEND_SCRIPT:-$SCRIPT_DIR/../data/synth_cpt_1m_pruned_v2/run_value_model_append_partial.sh}
SCOUT_VALUE_TAG=${SCOUT_VALUE_TAG:-v54eqdistance_pair_bonus_pairwise_currentref_solvedonly_timeoutfb${SCOUT_TIMEOUT_BEAM_FALLBACK_LIMIT}_secondary_v1}
SCOUT_VALUE_OUT_DIR=${SCOUT_VALUE_OUT_DIR:-outputs/candidate_value_model_${SCOUT_VALUE_TAG}}
BASE_VALUE_DATA=${BASE_VALUE_DATA:-outputs/candidate_value_model_v16_pairwise_solved_biased_progress_filter_oldfull_current4_v1/candidate_value_data.jsonl}
SCOUT_VALUE_DISABLE_PROGRESS_POSITIVES=${SCOUT_VALUE_DISABLE_PROGRESS_POSITIVES:-1}
SCOUT_VALUE_TRAIN_EXTRA_ARGS=${SCOUT_VALUE_TRAIN_EXTRA_ARGS:-}
if [ -z "$SCOUT_VALUE_TRAIN_EXTRA_ARGS" ]; then
  SCOUT_VALUE_TRAIN_EXTRA_ARGS="--objective pairwise --train_valid_only --epochs 20 --lr 0.01 --pairwise_negatives_per_positive 16"
fi
case "$SCOUT_REFRESH_VALUE_ROLE" in
  primary|secondary|none) ;;
  *)
    echo "invalid SCOUT_REFRESH_VALUE_ROLE: $SCOUT_REFRESH_VALUE_ROLE" >&2
    exit 1
    ;;
esac

SCOUT_CANDIDATE_EVAL_LIMIT=${SCOUT_CANDIDATE_EVAL_LIMIT:-0}
SCOUT_CANDIDATE_DEPTH_EVAL_LIMIT=${SCOUT_CANDIDATE_DEPTH_EVAL_LIMIT:-16}
SCOUT_CANDIDATE_TEMPLATE_BACKFILL_EXTRA_SLOTS=${SCOUT_CANDIDATE_TEMPLATE_BACKFILL_EXTRA_SLOTS:-16}
SCOUT_CANDIDATE_DEPTH_TYPE_EVAL_CAP=${SCOUT_CANDIDATE_DEPTH_TYPE_EVAL_CAP:-4}
SCOUT_CANDIDATE_DEPTH_TEMPLATE_EVAL_SLOTS=${SCOUT_CANDIDATE_DEPTH_TEMPLATE_EVAL_SLOTS:-2}
SCOUT_CANDIDATE_ADAPTIVE_TYPE_PENALTY=${SCOUT_CANDIDATE_ADAPTIVE_TYPE_PENALTY:-1}
SCOUT_CANDIDATE_ADAPTIVE_TYPE_PENALTY_THRESHOLD=${SCOUT_CANDIDATE_ADAPTIVE_TYPE_PENALTY_THRESHOLD:-32}
SCOUT_CANDIDATE_ADAPTIVE_TYPE_PENALTY_WEIGHT=${SCOUT_CANDIDATE_ADAPTIVE_TYPE_PENALTY_WEIGHT:-0.35}
SCOUT_CANDIDATE_ADAPTIVE_TYPE_PENALTY_MAX=${SCOUT_CANDIDATE_ADAPTIVE_TYPE_PENALTY_MAX:-1.5}
SCOUT_CANDIDATE_ADAPTIVE_TYPE_PENALTY_REASONS=${SCOUT_CANDIDATE_ADAPTIVE_TYPE_PENALTY_REASONS:-point_too_close,point_too_far,point_already_exists,unknown_point}
SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_ANCHOR=${SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_ANCHOR:-1}
SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_DELTA=${SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_DELTA:-60}
SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_DELTA_FLOOR=${SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_DELTA_FLOOR:-25}
SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_ROOT_RATIO=${SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_ROOT_RATIO:-0.6}
SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_BASE=${SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_BASE:-2.0}
SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_WEIGHT=${SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_WEIGHT:-0.08}
SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_MAX=${SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_MAX:-3.2}
SCOUT_CANDIDATE_STATIC_PROGRESS_TYPE_BONUS=${SCOUT_CANDIDATE_STATIC_PROGRESS_TYPE_BONUS:-${CANDIDATE_STATIC_PROGRESS_TYPE_BONUS:-}}
SCOUT_CANDIDATE_DDAR_TIMEOUT=${SCOUT_CANDIDATE_DDAR_TIMEOUT:-160}
SCOUT_CANDIDATE_WALL_TIMEOUT=${SCOUT_CANDIDATE_WALL_TIMEOUT:-100}
SCOUT_CANDIDATE_SOFT_TIMEOUT_MARGIN_SEC=${SCOUT_CANDIDATE_SOFT_TIMEOUT_MARGIN_SEC:-5}
SCOUT_CANDIDATE_DDAR_WORKERS=${SCOUT_CANDIDATE_DDAR_WORKERS:-8}
SCOUT_CANDIDATE_BEAM_SCORE=${SCOUT_CANDIDATE_BEAM_SCORE:-rerank_plus_progress}
SCOUT_CANDIDATE_BEAM_PROGRESS_WEIGHT=${SCOUT_CANDIDATE_BEAM_PROGRESS_WEIGHT:-0.6}
SCOUT_CANDIDATE_BEAM_PROGRESS_CAP=${SCOUT_CANDIDATE_BEAM_PROGRESS_CAP:-4.0}
SCOUT_CANDIDATE_DECODE_BEAM_LIMIT=${SCOUT_CANDIDATE_DECODE_BEAM_LIMIT:-16}
SCOUT_CANDIDATE_PROMPT_SAMPLING=${SCOUT_CANDIDATE_PROMPT_SAMPLING:-mixed_progress_constructive}
SCOUT_CANDIDATE_PROMPT_PREFERRED_TYPE_LIMIT=${SCOUT_CANDIDATE_PROMPT_PREFERRED_TYPE_LIMIT:-12}
SCOUT_CANDIDATE_TEMPLATE_PREFERRED_TYPE_LIMIT=${SCOUT_CANDIDATE_TEMPLATE_PREFERRED_TYPE_LIMIT:-36}
SCOUT_LM_FACT_CONTEXT_TOP_K=${SCOUT_LM_FACT_CONTEXT_TOP_K:-12}
SCOUT_BEAM_SIZE=${SCOUT_BEAM_SIZE:-64}
SCOUT_SEARCH_DEPTH=${SCOUT_SEARCH_DEPTH:-4}
SCOUT_NUM_RETURN_SEQUENCES=${SCOUT_NUM_RETURN_SEQUENCES:-48}
SCOUT_CANDIDATE_QUALITY_MULTIPLIER=${SCOUT_CANDIDATE_QUALITY_MULTIPLIER:-3}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "$(dirname "$SCOUT_LOG")" "$(dirname "$SCOUT_QUEUE_LOG")"

log() {
  date '+%F %T %z' | tr -d '\n' | tee -a "$SCOUT_QUEUE_LOG"
  printf ' %s\n' "$*" | tee -a "$SCOUT_QUEUE_LOG"
}

summary_rows() {
  python - "$WAIT_SUMMARY_JSONL" <<'PY'
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
if not path.exists():
    print(0)
else:
    print(sum(1 for line in path.read_text(encoding='utf-8', errors='replace').splitlines() if line.strip()))
PY
}

reference_process_active() {
  python - "$WAIT_PROCESS_PATTERN" <<'PY'
import os
import re
import subprocess
import sys

pattern = sys.argv[1]
self_pid = os.getpid()
parent_pid = os.getppid()
exclude_fragments = (
    'run_pairwise_scout_after_clean_wait.sh',
    'qwen_pairwise_scout_test.sh',
)
try:
    regex = re.compile(pattern)
except re.error:
    regex = re.compile(re.escape(pattern))
for line in subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True).splitlines():
    line = line.strip()
    if not line:
        continue
    pid_text, _, args = line.partition(' ')
    try:
        pid = int(pid_text)
    except ValueError:
        continue
    if pid in {self_pid, parent_pid}:
        continue
    if any(fragment in args for fragment in exclude_fragments):
        continue
    if regex.search(args):
        sys.exit(0)
sys.exit(1)
PY
}

log "waiting for reference clean rerun: $WAIT_OUT_DIR"
while true; do
  rows=$(summary_rows)
  if ! reference_process_active; then
    if [ "$rows" -ge "$WAIT_EXPECTED_ROWS" ] || { [ "$WAIT_ALLOW_INCOMPLETE" = "1" ] && [ "$rows" -ge "$WAIT_MIN_ROWS" ]; }; then
      break
    fi
    log "reference process ended with only ${rows}/${WAIT_EXPECTED_ROWS} rows"
    exit 1
  fi
  log "reference still active; summary rows=${rows}/${WAIT_EXPECTED_ROWS}"
  sleep "$WAIT_INTERVAL"
done

if [ -z "$SCOUT_PROBLEM_NAMES" ]; then
  SCOUT_PROBLEM_NAMES=$(python - "$WAIT_SUMMARY_JSONL" <<'PY'
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
names = []
for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    name = row.get('problem') or row.get('name')
    if name and not row.get('solved'):
        names.append(name)
print(','.join(names))
PY
)
fi

if [ -z "$SCOUT_PROBLEM_NAMES" ]; then
  log "no unsolved problems left after reference run; scout skipped"
  exit 0
fi

if [ -e "$SCOUT_OUT_DIR/summary.jsonl" ]; then
  echo "scout output already exists: $SCOUT_OUT_DIR" | tee -a "$SCOUT_QUEUE_LOG" >&2
  exit 1
fi

if [ "$TRAIN_SCOUT_VALUE_MODEL" = "1" ]; then
  log "training refreshed scout value model: $SCOUT_VALUE_TAG; role=$SCOUT_REFRESH_VALUE_ROLE"
  env \
    SCRIPT_DIR="$SCRIPT_DIR" \
    VALUE_TAG="$SCOUT_VALUE_TAG" \
    OUT_DIR="$SCOUT_VALUE_OUT_DIR" \
    BASE_VALUE_DATA="$BASE_VALUE_DATA" \
    PARTIAL_TAG="$(basename "$WAIT_OUT_DIR" | sed 's/^final_eval_imo_ag30_qwen_//')" \
    PARTIAL_OUT_DIR="$WAIT_OUT_DIR" \
    PARTIAL_EVENTS_DIR="$WAIT_OUT_DIR/events" \
    PARTIAL_SUMMARY_JSONL="$WAIT_SUMMARY_JSONL" \
    VALUE_DISABLE_PROGRESS_POSITIVES="$SCOUT_VALUE_DISABLE_PROGRESS_POSITIVES" \
    VALUE_TRAIN_EXTRA_ARGS="$SCOUT_VALUE_TRAIN_EXTRA_ARGS" \
    bash "$VALUE_MODEL_APPEND_SCRIPT" \
    >> "$SCOUT_QUEUE_LOG" 2>&1
  REFRESHED_SCOUT_VALUE_MODEL="$SCOUT_VALUE_OUT_DIR/candidate_value_model.json"
  case "$SCOUT_REFRESH_VALUE_ROLE" in
    primary)
      VALUE_MODEL="$REFRESHED_SCOUT_VALUE_MODEL"
      ;;
    secondary)
      SECONDARY_VALUE_MODEL="$REFRESHED_SCOUT_VALUE_MODEL"
      ;;
    none)
      ;;
  esac
  log "refreshed scout value model ready: $REFRESHED_SCOUT_VALUE_MODEL; value_model=$VALUE_MODEL; secondary_value_model=${SECONDARY_VALUE_MODEL:-none}"
fi

if [ ! -s "$VALUE_MODEL" ]; then
  echo "missing VALUE_MODEL: $VALUE_MODEL" | tee -a "$SCOUT_QUEUE_LOG" >&2
  exit 1
fi
if [ -n "$SECONDARY_VALUE_MODEL" ] && [ ! -s "$SECONDARY_VALUE_MODEL" ]; then
  echo "missing SECONDARY_VALUE_MODEL: $SECONDARY_VALUE_MODEL" | tee -a "$SCOUT_QUEUE_LOG" >&2
  exit 1
fi
if [ ! -s "$ADAPTER_PATH/adapter_model.safetensors" ]; then
  echo "missing adapter: $ADAPTER_PATH" | tee -a "$SCOUT_QUEUE_LOG" >&2
  exit 1
fi

log "starting pairwise scout: $SCOUT_TAG"
log "problem_names=$SCOUT_PROBLEM_NAMES"
log "depth_eval_limit=${SCOUT_CANDIDATE_DEPTH_EVAL_LIMIT}; template_backfill_extra_slots=${SCOUT_CANDIDATE_TEMPLATE_BACKFILL_EXTRA_SLOTS}; depth_type_cap=${SCOUT_CANDIDATE_DEPTH_TYPE_EVAL_CAP}; depth_template_slots=${SCOUT_CANDIDATE_DEPTH_TEMPLATE_EVAL_SLOTS}; adaptive_type_penalty=${SCOUT_CANDIDATE_ADAPTIVE_TYPE_PENALTY}/thr${SCOUT_CANDIDATE_ADAPTIVE_TYPE_PENALTY_THRESHOLD}/w${SCOUT_CANDIDATE_ADAPTIVE_TYPE_PENALTY_WEIGHT}/max${SCOUT_CANDIDATE_ADAPTIVE_TYPE_PENALTY_MAX}; dynamic_progress_type_anchor=${SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_ANCHOR}/min${SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_DELTA}/floor${SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_DELTA_FLOOR}/ratio${SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_ROOT_RATIO}/base${SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_BASE}/w${SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_WEIGHT}/max${SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_MAX}; static_progress_type_bonus=${SCOUT_CANDIDATE_STATIC_PROGRESS_TYPE_BONUS:-none}; depth_tail_slots=${SCOUT_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS}; depth_tail_strategy=${SCOUT_CANDIDATE_DEPTH_TAIL_EVAL_STRATEGY}; decode_beam_limit=${SCOUT_CANDIDATE_DECODE_BEAM_LIMIT}; candidate_timeout=${SCOUT_CANDIDATE_DDAR_TIMEOUT}; wall_timeout=${SCOUT_CANDIDATE_WALL_TIMEOUT}; soft_timeout_margin=${SCOUT_CANDIDATE_SOFT_TIMEOUT_MARGIN_SEC}; workers=${SCOUT_CANDIDATE_DDAR_WORKERS}; beam_score=${SCOUT_CANDIDATE_BEAM_SCORE}; progress_weight=${SCOUT_CANDIDATE_BEAM_PROGRESS_WEIGHT}; progress_cap=${SCOUT_CANDIDATE_BEAM_PROGRESS_CAP}; timeout_beam_fallback=${SCOUT_TIMEOUT_BEAM_FALLBACK_LIMIT}; timeout_fallback_mode=${SCOUT_TIMEOUT_BEAM_FALLBACK_MODE}; prompt_sampling=${SCOUT_CANDIDATE_PROMPT_SAMPLING}; prompt_preferred_type_limit=${SCOUT_CANDIDATE_PROMPT_PREFERRED_TYPE_LIMIT}; template_preferred_type_limit=${SCOUT_CANDIDATE_TEMPLATE_PREFERRED_TYPE_LIMIT}; lm_fact_top_k=${SCOUT_LM_FACT_CONTEXT_TOP_K}; rerank=${SCOUT_RERANK}; frontfill=${SCOUT_FRONTFILL_LIMIT}; value_model=$VALUE_MODEL; secondary_value_model=$SECONDARY_VALUE_MODEL; refresh_value_role=$SCOUT_REFRESH_VALUE_ROLE; value_disable_progress_positives=$SCOUT_VALUE_DISABLE_PROGRESS_POSITIVES"

if [ "$DRY_RUN" = "1" ]; then
  log "dry run enabled; scout command not launched"
  exit 0
fi

SECONDARY_VALUE_MODEL_ARGS=()
if [ -n "$SECONDARY_VALUE_MODEL" ]; then
  SECONDARY_VALUE_MODEL_ARGS=(
    --candidate_secondary_value_model "$SECONDARY_VALUE_MODEL"
    --candidate_frontfill_limit "$SCOUT_FRONTFILL_LIMIT"
  )
fi
ADAPTIVE_TYPE_PENALTY_ARGS=()
if [ "$SCOUT_CANDIDATE_ADAPTIVE_TYPE_PENALTY" = "1" ]; then
  ADAPTIVE_TYPE_PENALTY_ARGS=(
    --candidate_adaptive_type_penalty
    --candidate_adaptive_type_penalty_threshold "$SCOUT_CANDIDATE_ADAPTIVE_TYPE_PENALTY_THRESHOLD"
    --candidate_adaptive_type_penalty_weight "$SCOUT_CANDIDATE_ADAPTIVE_TYPE_PENALTY_WEIGHT"
    --candidate_adaptive_type_penalty_max "$SCOUT_CANDIDATE_ADAPTIVE_TYPE_PENALTY_MAX"
    --candidate_adaptive_type_penalty_reasons "$SCOUT_CANDIDATE_ADAPTIVE_TYPE_PENALTY_REASONS"
  )
fi
DYNAMIC_PROGRESS_TYPE_ANCHOR_ARGS=()
if [ "$SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_ANCHOR" = "1" ]; then
  DYNAMIC_PROGRESS_TYPE_ANCHOR_ARGS=(
    --candidate_dynamic_progress_type_anchor
    --candidate_dynamic_progress_type_min_delta "$SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_DELTA"
    --candidate_dynamic_progress_type_min_delta_floor "$SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_DELTA_FLOOR"
    --candidate_dynamic_progress_type_min_root_ratio "$SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_MIN_ROOT_RATIO"
    --candidate_dynamic_progress_type_bonus_base "$SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_BASE"
    --candidate_dynamic_progress_type_bonus_weight "$SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_WEIGHT"
    --candidate_dynamic_progress_type_bonus_max "$SCOUT_CANDIDATE_DYNAMIC_PROGRESS_TYPE_BONUS_MAX"
  )
fi
STATIC_PROGRESS_TYPE_BONUS_ARGS=()
if [ -n "$SCOUT_CANDIDATE_STATIC_PROGRESS_TYPE_BONUS" ]; then
  if [ ! -s "$SCOUT_CANDIDATE_STATIC_PROGRESS_TYPE_BONUS" ]; then
    echo "missing SCOUT_CANDIDATE_STATIC_PROGRESS_TYPE_BONUS: $SCOUT_CANDIDATE_STATIC_PROGRESS_TYPE_BONUS" | tee -a "$SCOUT_QUEUE_LOG" >&2
    exit 1
  fi
  STATIC_PROGRESS_TYPE_BONUS_ARGS=(
    --candidate_static_progress_type_bonus "$SCOUT_CANDIDATE_STATIC_PROGRESS_TYPE_BONUS"
  )
fi

xvfb-run -a -s "-screen 0 1024x768x24" python -u "$SCRIPT_DIR/run_qwen_ag_benchmark.py" \
  --script_dir "$SCRIPT_DIR" \
  --ag_repo repos/alphageometry \
  --problems_file repos/alphageometry/imo_ag_30.txt \
  --defs_file repos/alphageometry/defs.txt \
  --rules_file repos/alphageometry/rules.txt \
  --out_dir "$SCOUT_OUT_DIR" \
  --mode qwen \
  --problem_names "$SCOUT_PROBLEM_NAMES" \
  --qwen_model "$QWEN_MODEL" \
  --adapter_path "$ADAPTER_PATH" \
  --dtype bf16 \
  --device_map cuda:0 \
  --root_max_level 1000 \
  --root_ddar_timeout 600 \
  --candidate_max_level 300 \
  --candidate_ddar_timeout "$SCOUT_CANDIDATE_DDAR_TIMEOUT" \
  --candidate_wall_timeout "$SCOUT_CANDIDATE_WALL_TIMEOUT" \
  --candidate_soft_timeout_margin_sec "$SCOUT_CANDIDATE_SOFT_TIMEOUT_MARGIN_SEC" \
  --candidate_eval_limit "$SCOUT_CANDIDATE_EVAL_LIMIT" \
  --candidate_depth_eval_limit "$SCOUT_CANDIDATE_DEPTH_EVAL_LIMIT" \
  --candidate_template_backfill_extra_slots "$SCOUT_CANDIDATE_TEMPLATE_BACKFILL_EXTRA_SLOTS" \
  --candidate_depth_type_eval_cap "$SCOUT_CANDIDATE_DEPTH_TYPE_EVAL_CAP" \
  --candidate_depth_template_eval_slots "$SCOUT_CANDIDATE_DEPTH_TEMPLATE_EVAL_SLOTS" \
  "${ADAPTIVE_TYPE_PENALTY_ARGS[@]}" \
  "${DYNAMIC_PROGRESS_TYPE_ANCHOR_ARGS[@]}" \
  "${STATIC_PROGRESS_TYPE_BONUS_ARGS[@]}" \
  --candidate_depth_tail_eval_slots "$SCOUT_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS" \
  --candidate_depth_tail_eval_strategy "$SCOUT_CANDIDATE_DEPTH_TAIL_EVAL_STRATEGY" \
  --candidate_timeout_beam_fallback_limit "$SCOUT_TIMEOUT_BEAM_FALLBACK_LIMIT" \
  --candidate_timeout_beam_fallback_mode "$SCOUT_TIMEOUT_BEAM_FALLBACK_MODE" \
  --beam_size "$SCOUT_BEAM_SIZE" \
  --search_depth "$SCOUT_SEARCH_DEPTH" \
  --num_return_sequences "$SCOUT_NUM_RETURN_SEQUENCES" \
  --max_new_tokens 64 \
  --temperature 0.8 \
  --top_p 0.95 \
  --candidate_quality_multiplier "$SCOUT_CANDIDATE_QUALITY_MULTIPLIER" \
  --candidate_dsl_filter \
  --candidate_dsl_token_mask \
  --candidate_point_repair \
  --candidate_point_mask \
  --candidate_canonical_dedup \
  --candidate_prompt_sampling "$SCOUT_CANDIDATE_PROMPT_SAMPLING" \
  --candidate_prompt_preferred_type_limit "$SCOUT_CANDIDATE_PROMPT_PREFERRED_TYPE_LIMIT" \
  --candidate_template_preferred_type_limit "$SCOUT_CANDIDATE_TEMPLATE_PREFERRED_TYPE_LIMIT" \
  --candidate_template_backfill \
  --candidate_rerank "$SCOUT_RERANK" \
  --candidate_value_model "$VALUE_MODEL" \
  "${SECONDARY_VALUE_MODEL_ARGS[@]}" \
  --candidate_beam_score "$SCOUT_CANDIDATE_BEAM_SCORE" \
  --candidate_beam_progress_weight "$SCOUT_CANDIDATE_BEAM_PROGRESS_WEIGHT" \
  --candidate_beam_progress_cap "$SCOUT_CANDIDATE_BEAM_PROGRESS_CAP" \
  --candidate_decode_beam_limit "$SCOUT_CANDIDATE_DECODE_BEAM_LIMIT" \
  --candidate_ddar_workers "$SCOUT_CANDIDATE_DDAR_WORKERS" \
  --lm_fact_context_top_k "$SCOUT_LM_FACT_CONTEXT_TOP_K" \
  >> "$SCOUT_LOG" 2>&1

log "pairwise scout finished: $SCOUT_TAG"
