#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/alphageometry_repro}
AG_DIR=${AG_DIR:-$ROOT/alphageometry_clean}
VENV=${VENV:-$ROOT/venv}
RESULTS_DIR=${RESULTS_DIR:-$ROOT/imo_ag30_full_b32_beam512_depth16_clean_v1}
LOG=${LOG:-$ROOT/ag1_full_b32_beam512_depth16_clean_v1.nohup.log}

MIN_FREE_MEM_MB=${MIN_FREE_MEM_MB:-28000}
POLL_SECONDS=${POLL_SECONDS:-120}
STABLE_POLLS=${STABLE_POLLS:-3}

BATCH_SIZE=${BATCH_SIZE:-32}
BEAM_SIZE=${BEAM_SIZE:-512}
SEARCH_DEPTH=${SEARCH_DEPTH:-16}
WORKERS=${WORKERS:-96}

echo "[ag1-full] waiting for GPU free: min_free=${MIN_FREE_MEM_MB}MB stable_polls=${STABLE_POLLS}" | tee -a "$LOG"

stable=0
while true; do
  free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  qwen_pids=$(pgrep -f 'run_qwen_ag_benchmark.py' | paste -sd, - || true)
  now=$(date -Is)
  echo "[ag1-full] $now free_mem_mb=${free_mem:-unknown} qwen_pids=${qwen_pids:-none} stable=$stable" | tee -a "$LOG"

  if [[ -n "${free_mem:-}" && "$free_mem" -ge "$MIN_FREE_MEM_MB" && -z "$qwen_pids" ]]; then
    stable=$((stable + 1))
    if [[ "$stable" -ge "$STABLE_POLLS" ]]; then
      break
    fi
  else
    stable=0
  fi
  sleep "$POLL_SECONDS"
done

cd "$AG_DIR"
source "$VENV/bin/activate"

MELIAD_PATH="$AG_DIR/meliad_lib/meliad"
export PYTHONPATH="${PYTHONPATH:-}:$MELIAD_PATH"
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-1}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}

python - <<'PY' | tee -a "$LOG"
import jax
print("[ag1-full] jax_backend", jax.default_backend())
print("[ag1-full] jax_devices", jax.devices())
PY

echo "[ag1-full] starting benchmark results_dir=$RESULTS_DIR" | tee -a "$LOG"

python run_imo_ag30_benchmark.py \
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
  --keep_failed_candidate_logs \
  2>&1 | tee -a "$LOG"
