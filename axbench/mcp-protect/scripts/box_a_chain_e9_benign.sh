#!/usr/bin/env bash
# Box A chain v2: waits for box_a_chain_done.flag (E7 finished), then runs E9 benign regression
# E9 tests v17 f=0, 0.7, 1.0 + v19 f=0.3 on benign-only prompts (mcp_tox attack_mode='benign')
set -uo pipefail
cd /root/mcp-protect/axbench
export PATH="$HOME/.local/bin:$PATH"
set -a; source /root/mcp-protect/.env; set +a
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?need key}"
export LOCAL_KEY="EMPTY"

echo "[$(date +%H:%M:%S)] waiting for /tmp/box_a_chain_done.flag (E7 finish)..."
while [ ! -f /tmp/box_a_chain_done.flag ]; do sleep 30; done
echo "[$(date +%H:%M:%S)] E7 done; starting E9 benign regression"

V17_DIR=/root/mcp-protect/axbench/axbench/outputs/mcp_hsteer_qwen3_8b_v17_terse
V19_DIR=/root/mcp-protect/axbench/axbench/outputs/mcp_hsteer_qwen3_8b_v19_nothink
VFEVAL=/root/mcp-protect/prime-envs/.venv/bin/vf-eval
ENV_DIR=/root/mcp-protect/prime-envs/environments
OUT_ROOT="axbench/outputs/eval/E9_benign_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT"
N=50

run_one() {
  local tag="$1" dump_dir="$2" factor="$3"
  local odir="$OUT_ROOT/$tag"
  mkdir -p "$odir"
  echo "[$tag] dir=$dump_dir factor=$factor at $(date +%H:%M:%S)"
  pkill -f serve_mcp_hypersteer 2>/dev/null; sleep 2
  pkill -9 -f serve_mcp_hypersteer 2>/dev/null; sleep 1
  HYPERSTEER_CONCEPT_ID=0 HYPERSTEER_FACTOR="$factor" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup uv run --no-sync python axbench/mcp-protect/serve_mcp_hypersteer.py \
      --dump-dir "$dump_dir" --port 8000 --host 0.0.0.0 \
      > "$odir/serve.log" 2>&1 &
  for i in {1..90}; do
    sleep 2
    if curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q hypersteer; then
      echo "[$tag] serve ready"; break
    fi
  done
  if ! curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q hypersteer; then
    echo "[$tag] ERROR serve not ready"; tail -30 "$odir/serve.log"; return 1
  fi
  echo "[$tag] vf-eval N=$N benign mode at $(date +%H:%M:%S)"
  "$VFEVAL" mcp_tox \
    --env-dir-path "$ENV_DIR" --model hypersteer-local \
    --api-base-url http://127.0.0.1:8000/v1 --api-key-var LOCAL_KEY \
    --num-examples "$N" --rollouts-per-example 1 \
    --max-concurrent 1 --max-tokens 2048 --temperature 0.3 \
    --save-results --output-dir "$odir" \
    --env-args '{"judge_model": "openai/gpt-5.4-nano", "judge_api_key_var": "OPENROUTER_API_KEY", "attack_mode": "benign"}' \
    > "$odir/vf-eval.log" 2>&1
  echo "[$tag] done at $(date +%H:%M:%S)"
  pkill -f serve_mcp_hypersteer 2>/dev/null; sleep 1
}

# 4 cells: v17 f=0/0.7/1.0 + v19 f=0.3 — all on benign prompts
run_one e9_v17_f0_benign    "$V17_DIR" 0.0
run_one e9_v17_f0p7_benign  "$V17_DIR" 0.7
run_one e9_v17_f1p0_benign  "$V17_DIR" 1.0
run_one e9_v19_f0p3_benign  "$V19_DIR" 0.3

echo "============== SCORE E9 =============="
uv run --no-sync python axbench/outputs/_eval_tools/score_mcp_tox.py "$OUT_ROOT"
echo COMPLETE > /tmp/e9_done.flag
echo "[$(date +%H:%M:%S)] E9 benign regression complete. Output: $OUT_ROOT"
