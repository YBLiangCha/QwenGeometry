#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:-/root/rivermind-data/qwen_ag_lm}
cd "$WORK"
. "$WORK/env.sh" 2>/dev/null || true
. "$WORK/venv/bin/activate" 2>/dev/null || true

SCRIPT_DIR=${SCRIPT_DIR:-scripts}
PIPELINE_DIR=${PIPELINE_DIR:-data/synth_cpt_1m_pruned_v2}
POSTRUN_TAG=${POSTRUN_TAG:-postv12_solvedbiased_hybrid_v22tail_v1}
QUEUE_LOG=${QUEUE_LOG:-outputs/${POSTRUN_TAG}.queue_after_existing.log}
WAIT_INTERVAL=${WAIT_INTERVAL:-300}
DRY_RUN=${DRY_RUN:-0}

BLOCKING_PATTERN=${BLOCKING_PATTERN:-/tmp/qwen_ag_scripts_c541dd4|/tmp/qwen_ag_scripts_postv12_adaptive_progress_coverage_v1|run_qwen_ag_benchmark.py|train_qwen_aux_lora.py}
SCOUT_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS=${SCOUT_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS:-4}
CLEAN_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS=${CLEAN_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS:-4}

mkdir -p "$(dirname "$QUEUE_LOG")"

log() {
  date '+%F %T %z' | tr -d '\n' | tee -a "$QUEUE_LOG"
  printf ' %s\n' "$*" | tee -a "$QUEUE_LOG"
}

process_active() {
  python - "$BLOCKING_PATTERN" <<'PY'
import os
import re
import subprocess
import sys

pattern = sys.argv[1]
self_pid = os.getpid()
parent_pid = os.getppid()
try:
    regex = re.compile(pattern)
except re.error:
    regex = re.compile(re.escape(pattern))

for line in subprocess.check_output(['ps', '-eo', 'pid=,ppid=,args='], text=True).splitlines():
    line = line.strip()
    if not line:
        continue
    parts = line.split(maxsplit=2)
    if len(parts) < 3:
        continue
    pid_text, ppid_text, args = parts
    try:
        pid = int(pid_text)
        ppid = int(ppid_text)
    except ValueError:
        continue
    if pid in {self_pid, parent_pid} or ppid in {self_pid, parent_pid}:
        continue
    if 'queue_v22tail_after_existing_qwen_pipelines.sh' in args:
        continue
    if regex.search(args):
        print(line)
        sys.exit(0)
sys.exit(1)
PY
}

log "queue v22tail after existing Qwen pipelines"
log "blocking pattern: $BLOCKING_PATTERN"
while process_active >> "$QUEUE_LOG" 2>&1; do
  log "blocking Qwen pipeline still active; sleeping ${WAIT_INTERVAL}s"
  sleep "$WAIT_INTERVAL"
done

log "no blocking Qwen pipeline active; launching v22tail scout and hybrid wrapper"
log "tail slots: scout=${SCOUT_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS}; clean=${CLEAN_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS}"

if [ "$DRY_RUN" = "1" ]; then
  log "dry run enabled; not launching"
  exit 0
fi

SCOUT_LAUNCH_LOG=${SCOUT_LAUNCH_LOG:-outputs/${POSTRUN_TAG}.scout.launch.log}
HYBRID_LAUNCH_LOG=${HYBRID_LAUNCH_LOG:-outputs/${POSTRUN_TAG}.hybrid.launch.log}

env \
  SCRIPT_DIR="$SCRIPT_DIR" \
  SCOUT_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS="$SCOUT_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS" \
  nohup bash "$PIPELINE_DIR/run_pairwise_scout_after_clean_wait.sh" \
  > "$SCOUT_LAUNCH_LOG" 2>&1 < /dev/null &
SCOUT_PID=$!

env \
  SCRIPT_DIR="$SCRIPT_DIR" \
  POSTRUN_TAG="$POSTRUN_TAG" \
  QUEUE_LOG="outputs/${POSTRUN_TAG}.hybrid.queue.log" \
  WAIT_FOR_SCOUT=1 \
  SCOUT_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS="$SCOUT_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS" \
  CLEAN_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS="$CLEAN_CANDIDATE_DEPTH_TAIL_EVAL_SLOTS" \
  nohup bash "$PIPELINE_DIR/run_postv12_solvedbiased_hybrid_after_wait.sh" \
  > "$HYBRID_LAUNCH_LOG" 2>&1 < /dev/null &
HYBRID_PID=$!

echo "$SCOUT_PID" > "outputs/${POSTRUN_TAG}.scout.pid"
echo "$HYBRID_PID" > "outputs/${POSTRUN_TAG}.hybrid.pid"
log "launched scout pid=${SCOUT_PID}; hybrid pid=${HYBRID_PID}"
