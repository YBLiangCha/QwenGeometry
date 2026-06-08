#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:-/root/rivermind-data/qwen_ag_lm}
. "$WORK/env.sh" 2>/dev/null || true
. "$WORK/venv/bin/activate" 2>/dev/null || true
cd "$WORK"

SCRIPT_DIR=${SCRIPT_DIR:-scripts}
POSTRUN_SCRIPT=${POSTRUN_SCRIPT:-$SCRIPT_DIR/../data/synth_cpt_1m_pruned_v2/run_postrun_candidate_signal_sft_and_clean_rerun.sh}

REFERENCE_TAG=${REFERENCE_TAG:-unsolved_factctx_promptaug_top8_candidate_signal_postrun_value_v12_default_v1_depth48_t240_w150_nrs48_qm3_sigrep4_blinedia_statededup_nodediv_dsltpl_combotpl_rarecombo_vprior_v1}
REFERENCE_OUT_DIR=${REFERENCE_OUT_DIR:-outputs/final_eval_imo_ag30_qwen_${REFERENCE_TAG}}
REFERENCE_SUMMARY_JSONL=${REFERENCE_SUMMARY_JSONL:-$REFERENCE_OUT_DIR/summary.jsonl}
REFERENCE_PROCESS_PATTERN=${REFERENCE_PROCESS_PATTERN:-run_qwen_ag_benchmark.py.*${REFERENCE_TAG}}
REFERENCE_EXPECTED_ROWS=${REFERENCE_EXPECTED_ROWS:-16}
REFERENCE_MIN_ROWS=${REFERENCE_MIN_ROWS:-1}
REFERENCE_ALLOW_INCOMPLETE=${REFERENCE_ALLOW_INCOMPLETE:-0}

WAIT_FOR_SCOUT=${WAIT_FOR_SCOUT:-1}
SCOUT_TAG=${SCOUT_TAG:-unsolved_factctx_promptaug_top8_hybrid_v16_front12_v12_scout_after_v12_depth16_t160_w100_nrs48_qm3_v1}
SCOUT_OUT_DIR=${SCOUT_OUT_DIR:-outputs/final_eval_imo_ag30_qwen_${SCOUT_TAG}}
SCOUT_SUMMARY_JSONL=${SCOUT_SUMMARY_JSONL:-$SCOUT_OUT_DIR/summary.jsonl}
SCOUT_PROCESS_PATTERN=${SCOUT_PROCESS_PATTERN:-run_pairwise_scout_after_clean_wait.sh|run_qwen_ag_benchmark.py.*${SCOUT_TAG}}

WAIT_INTERVAL=${WAIT_INTERVAL:-300}
QUEUE_LOG=${QUEUE_LOG:-outputs/postv12_solvedbiased_hybrid_after_wait_v1.queue.log}
DRY_RUN=${DRY_RUN:-0}

POSTRUN_TAG=${POSTRUN_TAG:-postv12_solvedbiased_hybrid_v1}
SFT_OUT=${SFT_OUT:-outputs/stage4_candidate_signal_solvedbiased_after_v12_${POSTRUN_TAG}}
BASE_ADAPTER=${BASE_ADAPTER:-outputs/stage3_candidate_signal_after_factctx_lora_qwen2_5_7b_candidate_signal_sft_unsolved_factctx_promptaug_top8_adapter_value_v5_grammar_semantic_v3_v1_postrun_value_v12_default_v1}

PREFERRED_VALUE_MODEL=${PREFERRED_VALUE_MODEL:-outputs/candidate_value_model_v12_logistic_preddar_nodup_semantic_v3_partial7events6summary_v1/candidate_value_model.json}
FALLBACK_VALUE_MODEL=${FALLBACK_VALUE_MODEL:-outputs/candidate_value_model_v16_pairwise_solved_biased_progress_filter_oldfull_current4_v1/candidate_value_model.json}
VALUE_MODEL=${VALUE_MODEL:-$PREFERRED_VALUE_MODEL}
CLEAN_SECONDARY_VALUE_MODEL=${CLEAN_SECONDARY_VALUE_MODEL:-outputs/candidate_value_model_v18_pairwise_postv12_solvedonly_timeoutfb4_v1/candidate_value_model.json}
FALLBACK_SECONDARY_VALUE_MODEL=${FALLBACK_SECONDARY_VALUE_MODEL:-outputs/candidate_value_model_v16_pairwise_solved_biased_progress_filter_oldfull_current4_v1/candidate_value_model.json}
CLEAN_CANDIDATE_RERANK=${CLEAN_CANDIDATE_RERANK:-value_model_frontfill_diverse}
CLEAN_FRONTFILL_LIMIT=${CLEAN_FRONTFILL_LIMIT:-12}
CLEAN_CANDIDATE_DEPTH_EVAL_LIMIT=${CLEAN_CANDIDATE_DEPTH_EVAL_LIMIT:-24}
CLEAN_TIMEOUT_BEAM_FALLBACK_LIMIT=${CLEAN_TIMEOUT_BEAM_FALLBACK_LIMIT:-4}
CLEAN_CANDIDATE_DDAR_TIMEOUT=${CLEAN_CANDIDATE_DDAR_TIMEOUT:-200}
CLEAN_CANDIDATE_WALL_TIMEOUT=${CLEAN_CANDIDATE_WALL_TIMEOUT:-120}
CLEAN_CANDIDATE_DDAR_WORKERS=${CLEAN_CANDIDATE_DDAR_WORKERS:-8}
CLEAN_CANDIDATE_BEAM_SCORE=${CLEAN_CANDIDATE_BEAM_SCORE:-rerank_score}
CLEAN_CANDIDATE_DECODE_BEAM_LIMIT=${CLEAN_CANDIDATE_DECODE_BEAM_LIMIT:-16}
CLEAN_RERUN_TAG=${CLEAN_RERUN_TAG:-unsolved_factctx_promptaug_top8_stage4_solvedbiased_postv12_hybrid_v12_front12_v18coverage_beamscore_rerank_decbeam${CLEAN_CANDIDATE_DECODE_BEAM_LIMIT}_depth24_t200_w120_nrs48_qm3_timeoutfb${CLEAN_TIMEOUT_BEAM_FALLBACK_LIMIT}_v1}

STAGE4_SIGNAL_MIN_PROGRESS_DELTA=${STAGE4_SIGNAL_MIN_PROGRESS_DELTA:-80}
STAGE4_SIGNAL_MAX_ELAPSED_SEC=${STAGE4_SIGNAL_MAX_ELAPSED_SEC:-90}
STAGE4_SIGNAL_MIN_PROGRESS_EFFICIENCY=${STAGE4_SIGNAL_MIN_PROGRESS_EFFICIENCY:-1.0}
STAGE4_SIGNAL_MAX_PROGRESS_ROWS_PER_PROBLEM=${STAGE4_SIGNAL_MAX_PROGRESS_ROWS_PER_PROBLEM:-8}
STAGE4_SIGNAL_MAX_PROGRESS_ROWS_PER_TYPE=${STAGE4_SIGNAL_MAX_PROGRESS_ROWS_PER_TYPE:-16}
STAGE4_SIGNAL_SOLVED_REPEAT=${STAGE4_SIGNAL_SOLVED_REPEAT:-32}

mkdir -p "$(dirname "$QUEUE_LOG")"

log() {
  date '+%F %T %z' | tr -d '\n' | tee -a "$QUEUE_LOG"
  printf ' %s\n' "$*" | tee -a "$QUEUE_LOG"
}

summary_rows() {
  python - "$1" <<'PY'
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
if not path.exists():
    print(0)
else:
    print(sum(1 for line in path.read_text(encoding='utf-8', errors='replace').splitlines() if line.strip()))
PY
}

process_active() {
  python - "$1" <<'PY'
import os
import re
import subprocess
import sys

pattern = sys.argv[1]
self_pid = os.getpid()
parent_pid = os.getppid()
exclude_fragments = (
    'run_postv12_solvedbiased_hybrid_after_wait.sh',
    'python - "$1"',
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

log "waiting for reference v12 clean: $REFERENCE_TAG"
while true; do
  rows=$(summary_rows "$REFERENCE_SUMMARY_JSONL")
  if ! process_active "$REFERENCE_PROCESS_PATTERN"; then
    if [ "$rows" -ge "$REFERENCE_EXPECTED_ROWS" ] || { [ "$REFERENCE_ALLOW_INCOMPLETE" = "1" ] && [ "$rows" -ge "$REFERENCE_MIN_ROWS" ]; }; then
      break
    fi
    log "reference process ended before expected rows: ${rows}/${REFERENCE_EXPECTED_ROWS}"
    exit 1
  fi
  log "reference still active; summary rows=${rows}/${REFERENCE_EXPECTED_ROWS}"
  sleep "$WAIT_INTERVAL"
done

if [ "$WAIT_FOR_SCOUT" = "1" ]; then
  log "waiting for hybrid scout to finish: $SCOUT_TAG"
  while process_active "$SCOUT_PROCESS_PATTERN"; do
    scout_rows=$(summary_rows "$SCOUT_SUMMARY_JSONL")
    log "hybrid scout still active or queued; summary rows=${scout_rows}"
    sleep "$WAIT_INTERVAL"
  done
fi

if [ ! -s "$VALUE_MODEL" ] && [ -s "$FALLBACK_VALUE_MODEL" ]; then
  log "preferred value model missing, falling back: $FALLBACK_VALUE_MODEL"
  VALUE_MODEL="$FALLBACK_VALUE_MODEL"
fi
if [ ! -s "$VALUE_MODEL" ]; then
  echo "missing VALUE_MODEL for post-v12 clean rerun: $VALUE_MODEL" | tee -a "$QUEUE_LOG" >&2
  exit 1
fi
if [ "$CLEAN_CANDIDATE_RERANK" = "value_model_frontfill_diverse" ] && [ -n "$CLEAN_SECONDARY_VALUE_MODEL" ] && [ ! -s "$CLEAN_SECONDARY_VALUE_MODEL" ]; then
  if [ -s "$FALLBACK_SECONDARY_VALUE_MODEL" ]; then
    log "secondary value model missing, falling back: $FALLBACK_SECONDARY_VALUE_MODEL"
    CLEAN_SECONDARY_VALUE_MODEL="$FALLBACK_SECONDARY_VALUE_MODEL"
  else
    echo "missing CLEAN_SECONDARY_VALUE_MODEL for post-v12 clean rerun: $CLEAN_SECONDARY_VALUE_MODEL" | tee -a "$QUEUE_LOG" >&2
    exit 1
  fi
fi

CLEAN_PROBLEM_NAMES=$(python - "$REFERENCE_SUMMARY_JSONL" "$SCOUT_SUMMARY_JSONL" <<'PY'
import json
import pathlib
import sys

reference_path = pathlib.Path(sys.argv[1])
scout_path = pathlib.Path(sys.argv[2])
unsolved = []
for line in reference_path.read_text(encoding='utf-8', errors='replace').splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    name = row.get('problem') or row.get('name')
    if name and not row.get('solved'):
        unsolved.append(name)
scout_solved = set()
if scout_path.exists():
    for line in scout_path.read_text(encoding='utf-8', errors='replace').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        name = row.get('problem') or row.get('name')
        if name and row.get('solved'):
            scout_solved.add(name)
remaining = [name for name in unsolved if name not in scout_solved]
print(','.join(remaining))
PY
)

if [ -z "$CLEAN_PROBLEM_NAMES" ]; then
  log "no remaining unsolved problems after reference/scout; post-v12 pipeline skipped"
  exit 0
fi

log "post-v12 remaining problem names: $CLEAN_PROBLEM_NAMES"
log "stage4 adapter: $SFT_OUT"
log "hybrid clean tag: $CLEAN_RERUN_TAG"
log "hybrid clean timeout beam fallback limit: $CLEAN_TIMEOUT_BEAM_FALLBACK_LIMIT"
log "hybrid clean candidate beam score: $CLEAN_CANDIDATE_BEAM_SCORE"
log "hybrid clean candidate decode beam limit: $CLEAN_CANDIDATE_DECODE_BEAM_LIMIT"
log "hybrid clean value models: primary=$VALUE_MODEL; secondary=${CLEAN_SECONDARY_VALUE_MODEL:-none}"
log "stage4 signal filters: min_delta=${STAGE4_SIGNAL_MIN_PROGRESS_DELTA}; max_elapsed=${STAGE4_SIGNAL_MAX_ELAPSED_SEC}; min_eff=${STAGE4_SIGNAL_MIN_PROGRESS_EFFICIENCY}; per_problem=${STAGE4_SIGNAL_MAX_PROGRESS_ROWS_PER_PROBLEM}; per_type=${STAGE4_SIGNAL_MAX_PROGRESS_ROWS_PER_TYPE}; solved_repeat=${STAGE4_SIGNAL_SOLVED_REPEAT}"

if [ "$DRY_RUN" = "1" ]; then
  log "dry run enabled; postrun script not launched"
  exit 0
fi

env \
  SCRIPT_DIR="$SCRIPT_DIR" \
  OLD_TAG="$REFERENCE_TAG" \
  OLD_OUT_DIR="$REFERENCE_OUT_DIR" \
  OLD_EVENTS_DIR="$REFERENCE_OUT_DIR/events" \
  OLD_SUMMARY_JSONL="$REFERENCE_SUMMARY_JSONL" \
  WAIT_FOR_OLD_BENCH=0 \
  POSTRUN_TAG="$POSTRUN_TAG" \
  BASE_ADAPTER="$BASE_ADAPTER" \
  SIGNAL_MIN_PROGRESS_DELTA="$STAGE4_SIGNAL_MIN_PROGRESS_DELTA" \
  SIGNAL_MAX_ELAPSED_SEC="$STAGE4_SIGNAL_MAX_ELAPSED_SEC" \
  SIGNAL_MIN_PROGRESS_EFFICIENCY="$STAGE4_SIGNAL_MIN_PROGRESS_EFFICIENCY" \
  SIGNAL_MAX_PROGRESS_ROWS_PER_PROBLEM="$STAGE4_SIGNAL_MAX_PROGRESS_ROWS_PER_PROBLEM" \
  SIGNAL_MAX_PROGRESS_ROWS_PER_TYPE="$STAGE4_SIGNAL_MAX_PROGRESS_ROWS_PER_TYPE" \
  SIGNAL_SOLVED_REPEAT="$STAGE4_SIGNAL_SOLVED_REPEAT" \
  SFT_OUT="$SFT_OUT" \
  RUN_CLEAN_RERUN=1 \
  CLEAN_PROBLEM_NAMES="$CLEAN_PROBLEM_NAMES" \
  CLEAN_RERUN_TAG="$CLEAN_RERUN_TAG" \
  CLEAN_CANDIDATE_RERANK="$CLEAN_CANDIDATE_RERANK" \
  CLEAN_CANDIDATE_DEPTH_EVAL_LIMIT="$CLEAN_CANDIDATE_DEPTH_EVAL_LIMIT" \
  CLEAN_TIMEOUT_BEAM_FALLBACK_LIMIT="$CLEAN_TIMEOUT_BEAM_FALLBACK_LIMIT" \
  CLEAN_CANDIDATE_DDAR_TIMEOUT="$CLEAN_CANDIDATE_DDAR_TIMEOUT" \
  CLEAN_CANDIDATE_WALL_TIMEOUT="$CLEAN_CANDIDATE_WALL_TIMEOUT" \
  CLEAN_CANDIDATE_DDAR_WORKERS="$CLEAN_CANDIDATE_DDAR_WORKERS" \
  CLEAN_CANDIDATE_BEAM_SCORE="$CLEAN_CANDIDATE_BEAM_SCORE" \
  CLEAN_CANDIDATE_DECODE_BEAM_LIMIT="$CLEAN_CANDIDATE_DECODE_BEAM_LIMIT" \
  CLEAN_FRONTFILL_LIMIT="$CLEAN_FRONTFILL_LIMIT" \
  CLEAN_SECONDARY_VALUE_MODEL="$CLEAN_SECONDARY_VALUE_MODEL" \
  VALUE_MODEL="$VALUE_MODEL" \
  bash "$POSTRUN_SCRIPT"

log "post-v12 solved-biased hybrid pipeline finished"
