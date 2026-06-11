#!/usr/bin/env bash
set -euo pipefail

QWEN_WORK=${QWEN_WORK:-/root/rivermind-data/qwen_ag_lm}
AG_ROOT=${AG_ROOT:-/root/alphageometry_repro}
AG_DIR=${AG_DIR:-$AG_ROOT/alphageometry_clean}
AG_VENV=${AG_VENV:-$AG_ROOT/venv}

TAG=${TAG:-ag1_prompt_sft_fact_value_unsolved5_v1}
SFT_WORKDIR=${SFT_WORKDIR:-$AG_ROOT/${TAG}_ckpt}
RESULTS_DIR=${RESULTS_DIR:-$AG_ROOT/imo_ag30_${TAG}}
LOG=${LOG:-$QWEN_WORK/outputs/${TAG}.nohup.log}
WAIT_INTERVAL=${WAIT_INTERVAL:-120}

PROBLEMS=${PROBLEMS:-translated_imo_2008_p1b,translated_imo_2008_p6,translated_imo_2011_p6,translated_imo_2019_p2,translated_imo_2021_p3}
QWEN_ACTIVE_PATTERN=${QWEN_ACTIVE_PATTERN:-run_qwen_ag_benchmark.py|train_qwen_aux_lora.py|run_postrun_candidate_signal_sft_and_clean_rerun.sh|qwen_ablation_target7}

TRAIN_FILE_1=${TRAIN_FILE_1:-$QWEN_WORK/data/staged_1m_pruned_v2/candidate_signal_sft_v12v56_goalaware_v71_v1/candidate_signal_mixed_train.jsonl}
TRAIN_FILE_2=${TRAIN_FILE_2:-$QWEN_WORK/data/staged_1m_pruned_v2/factctx_promptaug_top8_stage2max2000_v1/fact_context_mixed_train.jsonl}
TRAIN_FILE_3=${TRAIN_FILE_3:-$QWEN_WORK/data/staged_1m_pruned_v2/stage2_aux_sft_train.jsonl}
EVAL_FILE_1=${EVAL_FILE_1:-$QWEN_WORK/data/staged_1m_pruned_v2/candidate_signal_sft_v12v56_goalaware_v71_v1/candidate_signal_mixed_eval.jsonl}
EVAL_FILE_2=${EVAL_FILE_2:-$QWEN_WORK/data/staged_1m_pruned_v2/factctx_promptaug_top8_stage2max2000_v1/fact_context_mixed_eval.jsonl}
EVAL_FILE_3=${EVAL_FILE_3:-$QWEN_WORK/data/staged_1m_pruned_v2/stage2_aux_sft_eval.jsonl}

VALUE_MODEL=${VALUE_MODEL:-$QWEN_WORK/outputs/candidate_value_model_v12_logistic_preddar_nodup_semantic_v3_partial7events6summary_v1/candidate_value_model.json}
SECONDARY_VALUE_MODEL=${SECONDARY_VALUE_MODEL:-$QWEN_WORK/outputs/candidate_value_model_v64prompt_component_feedback_pairwise_currentref_solvedonly_timeoutfb2_secondary_v1/candidate_value_model.json}
STATIC_TYPE_BONUS=${STATIC_TYPE_BONUS:-$QWEN_WORK/outputs/candidate_static_progress_type_bonus_postv12_solvedbiased_hybrid_v64prompt_component_feedback_v1.json}

TRAIN_STEPS=${TRAIN_STEPS:-1200}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
TRAIN_SEQUENCE_LENGTH=${TRAIN_SEQUENCE_LENGTH:-1024}
TRAIN_LR_MULT=${TRAIN_LR_MULT:-0.03}
TRAIN_WARMUP_STEPS=${TRAIN_WARMUP_STEPS:-50}
TRAIN_CHECKPOINT_EVERY=${TRAIN_CHECKPOINT_EVERY:-300}

AG_BATCH_SIZE=${AG_BATCH_SIZE:-32}
AG_BEAM_SIZE=${AG_BEAM_SIZE:-512}
AG_SEARCH_DEPTH=${AG_SEARCH_DEPTH:-16}
AG_WORKERS=${AG_WORKERS:-96}
AG_PROBLEM_TIME_LIMIT_SEC=${AG_PROBLEM_TIME_LIMIT_SEC:-5400}
AG_FACT_TOP_K=${AG_FACT_TOP_K:-12}

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

for path in \
  "$AG_DIR/imo_ag_30.txt" \
  "$AG_DIR/defs.txt" \
  "$AG_DIR/rules.txt" \
  "$AG_DIR/ag_ckpt_vocab/geometry.757.model" \
  "$AG_DIR/ag_ckpt_vocab/checkpoint_10999999" \
  "$AG_DIR/run_imo_ag30_value_rerank_benchmark.py" \
  "$QWEN_WORK/scripts/train_ag1_prompt_sft.py" \
  "$TRAIN_FILE_1" "$TRAIN_FILE_2" "$TRAIN_FILE_3" \
  "$EVAL_FILE_1" "$EVAL_FILE_2" "$EVAL_FILE_3" \
  "$VALUE_MODEL" "$SECONDARY_VALUE_MODEL" "$STATIC_TYPE_BONUS"; do
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

if [ -z "$(find "$SFT_WORKDIR" -maxdepth 1 -name 'checkpoint_*' -print -quit 2>/dev/null || true)" ]; then
  log "starting AG1 prompt SFT workdir=$SFT_WORKDIR"
  python "$QWEN_WORK/scripts/train_ag1_prompt_sft.py" \
    --train_file "$TRAIN_FILE_1" \
    --train_file "$TRAIN_FILE_2" \
    --train_file "$TRAIN_FILE_3" \
    --eval_file "$EVAL_FILE_1" \
    --eval_file "$EVAL_FILE_2" \
    --eval_file "$EVAL_FILE_3" \
    --workdir "$SFT_WORKDIR" \
    --load_dir "$AG_DIR/ag_ckpt_vocab" \
    --vocab_path "$AG_DIR/ag_ckpt_vocab/geometry.757.model" \
    --meliad_path "$MELIAD_PATH" \
    --num_steps "$TRAIN_STEPS" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --sequence_length "$TRAIN_SEQUENCE_LENGTH" \
    --learning_rate_multiplier "$TRAIN_LR_MULT" \
    --warmup_steps "$TRAIN_WARMUP_STEPS" \
    --checkpoint_every_steps "$TRAIN_CHECKPOINT_EVERY" \
    2>&1 | tee -a "$LOG"
else
  log "SFT checkpoint already exists, skipping training: $SFT_WORKDIR"
fi

if [ -e "$RESULTS_DIR/summary.json" ]; then
  log "results already exist: $RESULTS_DIR/summary.json"
  exit 0
fi

PROBLEM_ARGS=()
IFS=',' read -r -a problem_array <<< "$PROBLEMS"
for problem in "${problem_array[@]}"; do
  if [ -n "$problem" ]; then
    PROBLEM_ARGS+=(--problem "$problem")
  fi
done

log "starting fine-tuned AG1-LM fact+value run tag=$TAG"
log "problems=$PROBLEMS"
log "results_dir=$RESULTS_DIR"

python run_imo_ag30_value_rerank_benchmark.py \
  --problems_file="$AG_DIR/imo_ag_30.txt" \
  --defs_file="$AG_DIR/defs.txt" \
  --rules_file="$AG_DIR/rules.txt" \
  --ckpt_path="$SFT_WORKDIR" \
  --vocab_path="$AG_DIR/ag_ckpt_vocab/geometry.757.model" \
  --meliad_path="$MELIAD_PATH" \
  --results_dir="$RESULTS_DIR" \
  --batch_size="$AG_BATCH_SIZE" \
  --beam_size="$AG_BEAM_SIZE" \
  --search_depth="$AG_SEARCH_DEPTH" \
  --workers="$AG_WORKERS" \
  --skip_ddar_prefilter \
  --skip_initial_ddar \
  --problem_time_limit_sec="$AG_PROBLEM_TIME_LIMIT_SEC" \
  --keep_failed_candidate_logs \
  --qwen_search_path="$QWEN_WORK/scripts" \
  --lm_fact_context_top_k "$AG_FACT_TOP_K" \
  --candidate_rerank=value_model_frontfill_progress_diverse \
  --candidate_value_model="$VALUE_MODEL" \
  --candidate_secondary_value_model="$SECONDARY_VALUE_MODEL" \
  --candidate_frontfill_limit=8 \
  --candidate_static_progress_type_bonus="$STATIC_TYPE_BONUS" \
  "${PROBLEM_ARGS[@]}" \
  2>&1 | tee -a "$LOG"

log "finished fine-tuned AG1-LM fact+value run tag=$TAG"
