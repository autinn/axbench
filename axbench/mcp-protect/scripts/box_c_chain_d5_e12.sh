#!/usr/bin/env bash
# Box C chain: waits for E10 done flag, then runs D5 (DiffMean direct vec) + E12 (v11_FIXED 4096)
set -uo pipefail
cd /home/ubuntu/mcp-protect/axbench
export PATH="$HOME/.local/bin:$PATH"
set -a; source /home/ubuntu/mcp-protect/.env; set +a
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?need key}"
export LOCAL_KEY="EMPTY"

echo "[$(date +%H:%M:%S)] waiting for /tmp/e10_done.flag..."
while [ ! -f /tmp/e10_done.flag ]; do sleep 30; done
echo "[$(date +%H:%M:%S)] E10 done; starting D5 + E12 chain"

# === D5: DiffMean direct steering vec ===
# Uses v17's training data (input + output rows) to compute mean(positives_acts) − mean(negatives_acts)
# at L20 of un-steered Qwen3-8B, install as fixed steering, eval mcp_tox.

V17_DIR=/home/ubuntu/mcp-protect/axbench/axbench/outputs/mcp_hsteer_qwen3_8b_v17_terse
DIFFMEAN_DIR=/home/ubuntu/mcp-protect/axbench/axbench/outputs/diffmean_v17_L20
mkdir -p $DIFFMEAN_DIR

# Step 1: extract DiffMean vec from v17 training data
echo "[$(date +%H:%M:%S)] extracting DiffMean vec from v17 training data"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run --no-sync python - << 'PY'
import json, torch, pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

V17_PARQUET = "/home/ubuntu/mcp-protect/axbench/axbench/outputs/mcp_hsteer_qwen3_8b_v17_terse/generate/train_data.parquet"
OUT = "/home/ubuntu/mcp-protect/axbench/axbench/outputs/diffmean_v17_L20/diffmean_vec.pt"
LAYER = 20

print("loading model...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

df = pd.read_parquet(V17_PARQUET)
print(f"v17 rows: {len(df)} pos: {(df.category=='positive').sum()} neg: {(df.category=='negative').sum()}")

@torch.no_grad()
def get_act(text, layer=LAYER):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    outputs = model(**inputs, output_hidden_states=True)
    # Take last-token activation at layer
    hidden = outputs.hidden_states[layer]  # (1, seq, dim)
    return hidden[0, -1].cpu().float()  # (dim,)

print("extracting positive activations...")
pos_acts = []
for i, row in df[df.category == "positive"].iterrows():
    text = (row.input or "") + (row.output or "")
    pos_acts.append(get_act(text))
    if len(pos_acts) % 20 == 0:
        print(f"  pos {len(pos_acts)}")
pos_acts = torch.stack(pos_acts)
print(f"pos shape: {pos_acts.shape}")

print("extracting negative activations...")
neg_acts = []
for i, row in df[df.category == "negative"].iterrows():
    text = (row.input or "") + (row.output or "")
    neg_acts.append(get_act(text))
    if len(neg_acts) % 20 == 0:
        print(f"  neg {len(neg_acts)}")
if len(neg_acts) > 0:
    neg_acts = torch.stack(neg_acts)
    print(f"neg shape: {neg_acts.shape}")
    diffmean = pos_acts.mean(0) - neg_acts.mean(0)
else:
    print("WARN: no negatives in v17 data, using positives mean as direction")
    diffmean = pos_acts.mean(0)

print(f"diffmean shape: {diffmean.shape}, norm: {diffmean.norm().item():.4f}")
torch.save({"vec": diffmean, "layer": LAYER, "n_pos": len(pos_acts), "n_neg": len(neg_acts) if isinstance(neg_acts, torch.Tensor) else 0}, OUT)
print(f"wrote {OUT}")
PY
echo "[$(date +%H:%M:%S)] DiffMean vec extracted"

# Step 2: serve diffmean steering — this requires a custom serve. SKIP for now.
# Use existing diffmean infra in diffmean/serve.py if present.
ls /home/ubuntu/mcp-protect/diffmean/serve.py 2>&1 && echo "diffmean/serve.py present" || echo "diffmean/serve.py MISSING — D5 deferred (need custom diffmean serve)"

echo "D5_PARTIAL_DONE" > /tmp/d5_partial_done.flag
echo "[$(date +%H:%M:%S)] D5 partial complete (vec extracted; eval needs custom serve — deferred)"

# === E12: v11_FIXED at max_tokens=4096 thinking-ON ===
V11_DIR=/home/ubuntu/mcp-protect/axbench/axbench/outputs/mcp_hsteer_qwen3_8b_v11_audit
[ -d "$V11_DIR/train/hyperreft" ] || { echo "E12 SKIPPED: v11_FIXED weights not on box"; echo COMPLETE > /tmp/box_c_chain_done.flag; exit 0; }

VFEVAL=/home/ubuntu/mcp-protect/prime-envs/.venv/bin/vf-eval
ENV_DIR=/home/ubuntu/mcp-protect/prime-envs/environments
OUT_ROOT="axbench/outputs/eval/E12_v11FIXED_4096_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT"

run_one() {
  local tag="$1" factor="$2"
  local odir="$OUT_ROOT/$tag"
  mkdir -p "$odir"
  echo "[$tag] cid=0 factor=$factor at $(date +%H:%M:%S)"
  pkill -f serve_mcp_hypersteer 2>/dev/null; sleep 2
  pkill -9 -f serve_mcp_hypersteer 2>/dev/null; sleep 1
  HYPERSTEER_CONCEPT_ID=0 HYPERSTEER_FACTOR="$factor" HYPERSTEER_MAX_TOKENS=4096 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup uv run --no-sync python axbench/mcp-protect/serve_mcp_hypersteer.py \
      --dump-dir "$V11_DIR" --port 8000 --host 0.0.0.0 \
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
  echo "[$tag] vf-eval N=20 max_tokens=4096 at $(date +%H:%M:%S)"
  "$VFEVAL" mcp_tox \
    --env-dir-path "$ENV_DIR" --model hypersteer-local \
    --api-base-url http://127.0.0.1:8000/v1 --api-key-var LOCAL_KEY \
    --num-examples 20 --rollouts-per-example 1 \
    --max-concurrent 1 --max-tokens 4096 --temperature 0.3 \
    --save-results --output-dir "$odir" \
    --env-args '{"judge_model": "openai/gpt-5.4-nano", "judge_api_key_var": "OPENROUTER_API_KEY"}' \
    > "$odir/vf-eval.log" 2>&1
  echo "[$tag] done at $(date +%H:%M:%S)"
  pkill -f serve_mcp_hypersteer 2>/dev/null; sleep 1
}

run_one e12_v11FIXED_f0p3 0.3
run_one e12_v11FIXED_f0p5 0.5
run_one e12_v11FIXED_f0p7 0.7

echo "============== SCORE E12 =============="
uv run --no-sync python axbench/outputs/_eval_tools/score_mcp_tox.py "$OUT_ROOT"
echo COMPLETE > /tmp/e12_done.flag
echo COMPLETE > /tmp/box_c_chain_done.flag
echo "[$(date +%H:%M:%S)] Box C chain complete. Output: $OUT_ROOT"
