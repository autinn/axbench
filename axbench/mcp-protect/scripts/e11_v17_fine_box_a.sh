#!/usr/bin/env bash
# E11 v17 fine factor sweep on Box A (A40)
# Sweep around v17 f=0.7 (the regraded sweet spot at 0.86 AR)
set -uo pipefail
cd /root/mcp-protect/axbench
export PATH="$HOME/.local/bin:$PATH"
set -a; source /root/mcp-protect/.env; set +a
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?need key}"
export LOCAL_KEY="EMPTY"

V17_DIR=/root/mcp-protect/axbench/axbench/outputs/mcp_hsteer_qwen3_8b_v17_terse
VFEVAL=/root/mcp-protect/prime-envs/.venv/bin/vf-eval
ENV_DIR=/root/mcp-protect/prime-envs/environments
OUT_ROOT="axbench/outputs/eval/E11_v17_fine_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT"
N=20

run_one() {
  local tag="$1" factor="$2"
  local odir="$OUT_ROOT/$tag"
  mkdir -p "$odir"
  echo "[$tag] cid=0 factor=$factor at $(date +%H:%M:%S)"
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
  if ! curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q hypersteer; then
    echo "[$tag] ERROR serve not ready"; tail -30 "$odir/serve.log"; return 1
  fi
  echo "[$tag] vf-eval N=$N starting at $(date +%H:%M:%S)"
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

run_one e11_v17_f0p60  0.60
run_one e11_v17_f0p65  0.65
run_one e11_v17_f0p75  0.75
run_one e11_v17_f0p80  0.80

echo "============== SCORE ALL =============="
uv run --no-sync python axbench/outputs/_eval_tools/score_mcp_tox.py "$OUT_ROOT"
echo COMPLETE > /tmp/e11_v17_fine_done.flag
echo "Done. Output dir: $OUT_ROOT"
