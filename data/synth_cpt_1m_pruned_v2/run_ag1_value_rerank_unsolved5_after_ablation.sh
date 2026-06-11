#!/usr/bin/env bash
set -euo pipefail

QWEN_WORK=${QWEN_WORK:-/root/rivermind-data/qwen_ag_lm}
AG_ROOT=${AG_ROOT:-/root/alphageometry_repro}
AG_DIR=${AG_DIR:-$AG_ROOT/alphageometry_clean}
AG_VENV=${AG_VENV:-$AG_ROOT/venv}

TAG=${TAG:-ag1lm_value_rerank_ag1_unsolved5_v1}
RESULTS_DIR=${RESULTS_DIR:-$AG_ROOT/imo_ag30_${TAG}}
LOG=${LOG:-$QWEN_WORK/outputs/${TAG}.nohup.log}
WAIT_INTERVAL=${WAIT_INTERVAL:-120}
PROBLEM_TIME_LIMIT_SEC=${PROBLEM_TIME_LIMIT_SEC:-5400}

PROBLEMS=${PROBLEMS:-translated_imo_2008_p1b,translated_imo_2008_p6,translated_imo_2011_p6,translated_imo_2019_p2,translated_imo_2021_p3}
QWEN_ACTIVE_PATTERN=${QWEN_ACTIVE_PATTERN:-run_qwen_ag_benchmark.py|train_qwen_aux_lora.py|run_postrun_candidate_signal_sft_and_clean_rerun.sh|qwen_ablation_target7}

VALUE_MODEL=${VALUE_MODEL:-$QWEN_WORK/outputs/candidate_value_model_v12_logistic_preddar_nodup_semantic_v3_partial7events6summary_v1/candidate_value_model.json}
SECONDARY_VALUE_MODEL=${SECONDARY_VALUE_MODEL:-$QWEN_WORK/outputs/candidate_value_model_v64prompt_component_feedback_pairwise_currentref_solvedonly_timeoutfb2_secondary_v1/candidate_value_model.json}
STATIC_TYPE_BONUS=${STATIC_TYPE_BONUS:-$QWEN_WORK/outputs/candidate_static_progress_type_bonus_postv12_solvedbiased_hybrid_v64prompt_component_feedback_v1.json}

BATCH_SIZE=${BATCH_SIZE:-32}
BEAM_SIZE=${BEAM_SIZE:-512}
SEARCH_DEPTH=${SEARCH_DEPTH:-16}
WORKERS=${WORKERS:-96}

mkdir -p "$(dirname "$LOG")"

log() {
  date '+%F %T %z' | tr -d '\n' | tee -a "$LOG"
  printf ' %s\n' "$*" | tee -a "$LOG"
}

log "waiting for Qwen ablation/bench processes to finish"
while pgrep -f "$QWEN_ACTIVE_PATTERN" >/dev/null; do
  pgrep -af "$QWEN_ACTIVE_PATTERN" | tee -a "$LOG" || true
  sleep "$WAIT_INTERVAL"
done

if [ -e "$RESULTS_DIR/summary.json" ]; then
  log "results already exist: $RESULTS_DIR/summary.json"
  exit 0
fi

for path in \
  "$AG_DIR/imo_ag_30.txt" \
  "$AG_DIR/defs.txt" \
  "$AG_DIR/rules.txt" \
  "$AG_DIR/ag_ckpt_vocab/geometry.757.model" \
  "$AG_DIR/ag_ckpt_vocab/checkpoint_10999999" \
  "$AG_DIR/run_imo_ag30_value_rerank_benchmark.py" \
  "$VALUE_MODEL" \
  "$SECONDARY_VALUE_MODEL" \
  "$STATIC_TYPE_BONUS"; do
  if [ ! -e "$path" ]; then
    log "missing required path: $path"
    exit 1
  fi
done

cd "$AG_DIR"
. "$AG_VENV/bin/activate"

MELIAD_PATH="$AG_DIR/meliad_lib/meliad"
export PYTHONPATH="$MELIAD_PATH:$QWEN_WORK/scripts:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-1}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}

PROBLEM_ARGS=()
IFS=',' read -r -a problem_array <<< "$PROBLEMS"
for problem in "${problem_array[@]}"; do
  if [ -n "$problem" ]; then
    PROBLEM_ARGS+=(--problem "$problem")
  fi
done

log "starting AG1-LM value-rerank experiment tag=$TAG"
log "problems=$PROBLEMS"
log "results_dir=$RESULTS_DIR"

python run_imo_ag30_value_rerank_benchmark.py \
  --problems_file="$AG_DIR/imo_ag_30.txt" \
  --defs_file="$AG_DIR/defs.txt" \
  --rules_file="$AG_DIR/rules.txt" \
  --ckpt_path="$AG_DIR/ag_ckpt_vocab" \
  --vocab_path="$AG_DIR/ag_ckpt_vocab/geometry.757.model" \
  --meliad_path="$MELIAD_PATH" \
  --results_dir="$RESULTS_DIR" \
  --batch_size="$BATCH_SIZE" \
  --beam_size="$BEAM_SIZE" \
  --search_depth="$SEARCH_DEPTH" \
  --workers="$WORKERS" \
  --skip_ddar_prefilter \
  --skip_initial_ddar \
  --problem_time_limit_sec="$PROBLEM_TIME_LIMIT_SEC" \
  --keep_failed_candidate_logs \
  --qwen_search_path="$QWEN_WORK/scripts" \
  --candidate_rerank=value_model_frontfill_progress_diverse \
  --candidate_value_model="$VALUE_MODEL" \
  --candidate_secondary_value_model="$SECONDARY_VALUE_MODEL" \
  --candidate_frontfill_limit=8 \
  --candidate_static_progress_type_bonus="$STATIC_TYPE_BONUS" \
  "${PROBLEM_ARGS[@]}" \
  2>&1 | tee -a "$LOG"

log "finished AG1-LM value-rerank experiment tag=$TAG"
