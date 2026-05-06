#!/usr/bin/env bash
# E11 v19 fine factor sweep on Box B (A100)
# Sweep around v19 f=0.3 (the regraded LOW-factor clean win at 0.72 AR)
# Wait until v19 weights have arrived (currently transferring from Box A)
set -uo pipefail
cd /home/ubuntu/mcp-protect/axbench
export PATH="$HOME/.local/bin:$PATH"
set -a; source /home/ubuntu/mcp-protect/.env; set +a
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?need key}"
export LOCAL_KEY="EMPTY"

V19_DIR=/home/ubuntu/mcp-protect/axbench/axbench/outputs/mcp_hsteer_qwen3_8b_v19_nothink
VFEVAL=/home/ubuntu/mcp-protect/prime-envs/.venv/bin/vf-eval
ENV_DIR=/home/ubuntu/mcp-protect/prime-envs/environments
OUT_ROOT="axbench/outputs/eval/E11_v19_fine_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT"
N=20

# Wait for v19 weights (currently transferring from Box A → Box B)
echo "[$(date +%H:%M:%S)] waiting for v19 weights..."
while [ ! -f "$V19_DIR/train/hyperreft/model.safetensors" ] || [ "$(stat -c %s "$V19_DIR/train/hyperreft/model.safetensors" 2>/dev/null || echo 0)" -lt 3000000000 ]; do
  sleep 15
done
echo "[$(date +%H:%M:%S)] v19 weights arrived"

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
      --dump-dir "$V19_DIR" --port 8000 --host 0.0.0.0 \
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

run_one e11_v19_f0p20  0.20
run_one e11_v19_f0p25  0.25
run_one e11_v19_f0p35  0.35
run_one e11_v19_f0p40  0.40
run_one e11_v19_f0p45  0.45

echo "============== SCORE ALL =============="
uv run --no-sync python axbench/outputs/_eval_tools/score_mcp_tox.py "$OUT_ROOT"
echo COMPLETE > /tmp/e11_v19_fine_done.flag
echo "Done. Output dir: $OUT_ROOT"
