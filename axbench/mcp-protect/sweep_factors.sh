#!/usr/bin/env bash
# sweep_factors.sh — restart serve_mcp_hypersteer at multiple FACTOR values,
# run vf-eval against mcp_tox each time, score with score_mcp_tox.py.
#
# Usage:
#   bash sweep_factors.sh <DUMP_DIR> <CONCEPT_ID> <N_SAMPLES> <FACTORS_CSV> [TAG]
# Example:
#   bash sweep_factors.sh axbench/outputs/mcp_hsteer_9b_v3 0 20 "0.0,1.0,2.0" v3-quickscan
set -uo pipefail

DUMP_DIR="${1:?dump dir}"
CID="${2:?concept_id}"
N="${3:?N samples}"
FACTORS_CSV="${4:?factors comma-separated}"
TAG="${5:-$(basename "$DUMP_DIR")}"

cd /home/ubuntu/mcp-protect/axbench
export PATH="$HOME/.local/bin:$PATH"
set -a; source /home/ubuntu/mcp-protect/.env; set +a
# OPENROUTER_API_KEY is what mcp_tox judge wants by default
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?need OPENROUTER_API_KEY in .env}"
# vf-eval --api-key-var expects an env var name; we point it at this one (any value works for our local serve)
export LOCAL_KEY="EMPTY"

VFEVAL=/home/ubuntu/mcp-protect/prime-envs/.venv/bin/vf-eval
ENV_DIR=/home/ubuntu/mcp-protect/prime-envs/environments

OUT_ROOT="axbench/outputs/eval/${TAG}"
mkdir -p "$OUT_ROOT"

kill_serve() {
  pkill -f "serve_mcp_hypersteer" 2>/dev/null
  sleep 2
  pkill -9 -f "serve_mcp_hypersteer" 2>/dev/null
  sleep 1
}

start_serve() {
  local factor="$1"
  echo "[sweep] starting serve dump=$DUMP_DIR cid=$CID factor=$factor" >&2
  HYPERSTEER_CONCEPT_ID="$CID" HYPERSTEER_FACTOR="$factor" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup uv run --no-sync python axbench/mcp-protect/serve_mcp_hypersteer.py \
      --dump-dir "$DUMP_DIR" --port 8000 --host 0.0.0.0 \
      > /tmp/serve_${TAG}_f${factor}.log 2>&1 &
  echo $! > /tmp/serve.pid
  for i in {1..60}; do
    sleep 2
    if curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q hypersteer; then
      echo "[sweep] serve ready after ${i}*2s" >&2
      return 0
    fi
  done
  echo "[sweep] ERROR serve not ready" >&2
  tail -30 /tmp/serve_${TAG}_f${factor}.log >&2
  return 1
}

run_eval() {
  local factor="$1"
  local odir="$OUT_ROOT/factor_${factor}"
  mkdir -p "$odir"
  echo "[sweep] vf-eval n=$N factor=$factor -> $odir" >&2
  "$VFEVAL" mcp_tox \
    --env-dir-path "$ENV_DIR" \
    --model hypersteer-local \
    --api-base-url http://127.0.0.1:8000/v1 \
    --api-key-var LOCAL_KEY \
    --num-examples "$N" \
    --rollouts-per-example 1 \
    --max-concurrent 4 \
    --max-tokens 512 \
    --temperature 0.3 \
    --save-results \
    --output-dir "$odir" \
    --debug \
    --env-args '{"judge_model": "openai/gpt-5.4-nano", "judge_api_key_var": "OPENROUTER_API_KEY"}' \
    > "$odir/vf-eval.log" 2>&1
}

IFS=',' read -ra FACTORS <<< "$FACTORS_CSV"
for f in "${FACTORS[@]}"; do
  kill_serve
  start_serve "$f" || { echo "[sweep] skip f=$f (serve failed)" >&2; continue; }
  run_eval "$f"
  kill_serve
done

echo
echo "=== SCORE ==="
uv run python axbench/outputs/_eval_tools/score_mcp_tox.py "$OUT_ROOT"
