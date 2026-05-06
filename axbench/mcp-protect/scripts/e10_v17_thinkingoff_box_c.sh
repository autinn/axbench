#!/usr/bin/env bash
# E10 — v17 f=0.7 with MCP_ENABLE_THINKING=0 on Box C (A6000)
# Tests if v17 f=0.7 stacks with thinking-off OR if its win is independent of thinking
set -uo pipefail
cd /home/ubuntu/mcp-protect/axbench
export PATH="$HOME/.local/bin:$PATH"
set -a; source /home/ubuntu/mcp-protect/.env; set +a
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?need key}"
export LOCAL_KEY="EMPTY"

V17_DIR=/home/ubuntu/mcp-protect/axbench/axbench/outputs/mcp_hsteer_qwen3_8b_v17_terse
VFEVAL=/home/ubuntu/mcp-protect/prime-envs/.venv/bin/vf-eval
ENV_DIR=/home/ubuntu/mcp-protect/prime-envs/environments
OUT_ROOT="axbench/outputs/eval/E10_v17_thinkOff_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT"
N=50

run_one() {
  local tag="$1" factor="$2" thinking="$3"
  local odir="$OUT_ROOT/$tag"
  mkdir -p "$odir"
  echo "[$tag] cid=0 factor=$factor thinking=$thinking at $(date +%H:%M:%S)"
  pkill -f serve_mcp_hypersteer 2>/dev/null; sleep 2
  pkill -9 -f serve_mcp_hypersteer 2>/dev/null; sleep 1
  HYPERSTEER_CONCEPT_ID=0 HYPERSTEER_FACTOR="$factor" \
    MCP_ENABLE_THINKING="$thinking" \
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
  if ! curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q hypersteer; then
    echo "[$tag] ERROR serve not ready"; tail -30 "$odir/serve.log"; return 1
  fi
  echo "[$tag] vf-eval N=$N at $(date +%H:%M:%S)"
  "$VFEVAL" mcp_tox \
    --env-dir-path "$ENV_DIR" --model hypersteer-local \
    --api-base-url http://127.0.0.1:8000/v1 --api-key-var LOCAL_KEY \
    --num-examples "$N" --rollouts-per-example 1 \
    --max-concurrent 1 --max-tokens 2048 --temperature 0.3 \
    --save-results --output-dir "$odir" \
    --env-args '{"judge_model": "openai/gpt-5.4-nano", "judge_api_key_var": "OPENROUTER_API_KEY"}' \
    > "$odir/vf-eval.log" 2>&1
  echo "[$tag] done at $(date +%H:%M:%S)"
  pkill -f serve_mcp_hypersteer 2>/dev/null; sleep 1
}

# Two cells: v17 f=0.7 thinking ON (control, should match prior 0.86) vs thinking OFF (the test)
run_one e10_v17_f0p7_thinkOn   0.7  1
run_one e10_v17_f0p7_thinkOff  0.7  0

# Also baseline thinking-OFF (no steering) so we can isolate the contribution
run_one e10_baseline_thinkOff  0.0  0

echo "============== SCORE ALL =============="
uv run --no-sync python axbench/outputs/_eval_tools/score_mcp_tox.py "$OUT_ROOT"
echo COMPLETE > /tmp/e10_done.flag
echo "Done. Output dir: $OUT_ROOT"
