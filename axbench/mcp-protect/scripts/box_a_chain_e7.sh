#!/usr/bin/env bash
# Box A chain: waits for E11-v17 done flag, then runs E7 (v17 f=1.0 replication on prompts 50-99)
set -uo pipefail
cd /root/mcp-protect/axbench
export PATH="$HOME/.local/bin:$PATH"
set -a; source /root/mcp-protect/.env; set +a
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?need key}"
export LOCAL_KEY="EMPTY"

echo "[$(date +%H:%M:%S)] waiting for /tmp/e11_v17_fine_done.flag..."
while [ ! -f /tmp/e11_v17_fine_done.flag ]; do sleep 30; done
echo "[$(date +%H:%M:%S)] E11-v17 done; starting E7"

# === E7: replicate v17 f=1.0 on a different prompt slice ===
# vf-eval doesn't natively support a prompt OFFSET, so we shuffle the dataset with a different
# random seed in the env-args. Use --rollouts-per-example 1 N=50 with seed=99 (current default seed=42).
V17_DIR=/root/mcp-protect/axbench/axbench/outputs/mcp_hsteer_qwen3_8b_v17_terse
VFEVAL=/root/mcp-protect/prime-envs/.venv/bin/vf-eval
ENV_DIR=/root/mcp-protect/prime-envs/environments
OUT_ROOT="axbench/outputs/eval/E7_v17_replicate_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT"

run_one() {
  local tag="$1" factor="$2" seed="$3"
  local odir="$OUT_ROOT/$tag"
  mkdir -p "$odir"
  echo "[$tag] cid=0 factor=$factor seed=$seed at $(date +%H:%M:%S)"
  pkill -f serve_mcp_hypersteer 2>/dev/null; sleep 2
  pkill -9 -f serve_mcp_hypersteer 2>/dev/null; sleep 1
  HYPERSTEER_CONCEPT_ID=0 HYPERSTEER_FACTOR="$factor" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup uv run --no-sync python axbench/mcp-protect/serve_mcp_hypersteer.py \
      --dump-dir "$V17_DIR" --port 8000 --host 0.0.0.0 \
      > "$odir/serve.log" 2>&1 &
  for i in {1..90}; do
    sleep 2
    if curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q hypersteer; then
      echo "[$tag] serve ready"; break
    fi
  done
  echo "[$tag] vf-eval N=50 seed=$seed at $(date +%H:%M:%S)"
  "$VFEVAL" mcp_tox \
    --env-dir-path "$ENV_DIR" --model hypersteer-local \
    --api-base-url http://127.0.0.1:8000/v1 --api-key-var LOCAL_KEY \
    --num-examples 50 --rollouts-per-example 1 \
    --max-concurrent 1 --max-tokens 2048 --temperature 0.3 \
    --save-results --output-dir "$odir" \
    --env-args "{\"judge_model\": \"openai/gpt-5.4-nano\", \"judge_api_key_var\": \"OPENROUTER_API_KEY\"}" \
    > "$odir/vf-eval.log" 2>&1
  echo "[$tag] done at $(date +%H:%M:%S)"
  pkill -f serve_mcp_hypersteer 2>/dev/null; sleep 1
}

# Replicate v17 f=1.0 with a different seed — vf-eval shuffle order changes the prompts seen
# Also do f=0.7 for the same different-seed set, for comparison
run_one e7_v17_f1p0_seed99 1.0 99
run_one e7_v17_f0p7_seed99 0.7 99
run_one e7_v17_f0_seed99   0.0 99

echo "============== SCORE E7 =============="
uv run --no-sync python axbench/outputs/_eval_tools/score_mcp_tox.py "$OUT_ROOT"
echo COMPLETE > /tmp/e7_done.flag
echo COMPLETE > /tmp/box_a_chain_done.flag
echo "[$(date +%H:%M:%S)] Box A chain done. Output: $OUT_ROOT"
