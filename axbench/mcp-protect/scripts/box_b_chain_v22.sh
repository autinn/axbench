#!/usr/bin/env bash
# Box B chain: waits for E11-v19 done flag, then trains v22 multi-concept HyperSteer + evals
set -uo pipefail
cd /home/ubuntu/mcp-protect/axbench
export PATH="$HOME/.local/bin:$PATH"
set -a; source /home/ubuntu/mcp-protect/.env; set +a
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?need key}"
export LOCAL_KEY="EMPTY"

echo "[$(date +%H:%M:%S)] waiting for /tmp/e11_v19_fine_done.flag..."
while [ ! -f /tmp/e11_v19_fine_done.flag ]; do sleep 30; done
echo "[$(date +%H:%M:%S)] E11-v19 done; starting v22 train"

# === v22 multi-concept HyperSteer ===
# Uses 200-row v19-style dataset (12 archetypes × ~17 rows). Single-concept training on each cid.
V22_DIR=/home/ubuntu/mcp-protect/axbench/axbench/outputs/mcp_hsteer_qwen3_8b_v22_multi
mkdir -p $V22_DIR/generate $V22_DIR/train

# Wait for v22 dataset to be uploaded
echo "[$(date +%H:%M:%S)] waiting for v22 dataset..."
while [ ! -f $V22_DIR/generate/train_data.parquet ]; do sleep 15; done
echo "[$(date +%H:%M:%S)] v22 dataset present"

cat > $V22_DIR/mcp_hypersteer_config.yaml << 'EOF'
generate:
  lm_model: openai/gpt-5.4-nano
  input_length: 32
  output_length: 32
  num_of_examples: 1
  concept_path: /home/ubuntu/mcp-protect/axbench/axbench/outputs/mcp_hsteer_qwen3_8b_v22_multi/merged_concepts_mcp.json
  dataset_category: instruction
  master_data_dir: axbench/data
  seed: 42
  keep_orig_axbench_format: true
train:
  model_name: Qwen/Qwen3-8B
  layer: 20
  component: res
  seed: 42
  use_bf16: true
  output_length: 2048
  max_concepts: 12
  models:
    HyperSteer:
      batch_size: 1
      gradient_accumulation_steps: 8
      n_epochs: 5
      lr: 2.0000e-05
      weight_decay: 0.0
      low_rank_dimension: 1
      intervention_positions: all
      intervention_type: addition
      binarize_dataset: false
      train_on_negative: false
      exclude_bos: true
      hypernet_name_or_path: Qwen/Qwen3-8B
      num_hidden_layers: 4
      hypernet_initialize_from_pretrained: true
      max_input_length: 2048
      max_concept_length: 1024
inference:
  use_bf16: true
  models:
    - HyperSteer
  model_name: Qwen/Qwen3-8B
  output_length: 2048
  steering_intervention_type: addition
  steering_model_name: Qwen/Qwen3-8B
  steering_layers:
    - 20
  master_data_dir: axbench/data
  seed: 42
  lm_model: openai/gpt-5.4-nano
  temperature: 1.0
EOF

echo "[$(date +%H:%M:%S)] === TRAIN v22 multi-concept ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run --no-sync torchrun --nproc_per_node=1 \
    axbench/scripts/train.py \
    --config $V22_DIR/mcp_hypersteer_config.yaml \
    --dump_dir $V22_DIR \
    2>&1 | tee /tmp/train_v22.log
echo "[$(date +%H:%M:%S)] === v22 TRAIN DONE ==="
ls $V22_DIR/train/hyperreft/ 2>&1 | head
[ -f $V22_DIR/train/hyperreft/model.safetensors ] || { echo "ERROR: weights not saved"; exit 1; }
echo COMPLETE > /tmp/v22_train_done.flag

# === eval v22 across 12 cids at f=0.3, 0.5 ===
VFEVAL=/home/ubuntu/mcp-protect/prime-envs/.venv/bin/vf-eval
ENV_DIR=/home/ubuntu/mcp-protect/prime-envs/environments
OUT_ROOT="axbench/outputs/eval/v22_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT"

run_one() {
  local tag="$1" cid="$2" factor="$3"
  local odir="$OUT_ROOT/$tag"
  mkdir -p "$odir"
  echo "[$tag] cid=$cid factor=$factor at $(date +%H:%M:%S)"
  pkill -f serve_mcp_hypersteer 2>/dev/null; sleep 2
  pkill -9 -f serve_mcp_hypersteer 2>/dev/null; sleep 1
  HYPERSTEER_CONCEPT_ID="$cid" HYPERSTEER_FACTOR="$factor" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup uv run --no-sync python axbench/mcp-protect/serve_mcp_hypersteer.py \
      --dump-dir "$V22_DIR" --port 8000 --host 0.0.0.0 \
      > "$odir/serve.log" 2>&1 &
  for i in {1..90}; do
    sleep 2
    if curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q hypersteer; then
      echo "[$tag] serve ready"; break
    fi
  done
  echo "[$tag] vf-eval N=20 at $(date +%H:%M:%S)"
  "$VFEVAL" mcp_tox \
    --env-dir-path "$ENV_DIR" --model hypersteer-local \
    --api-base-url http://127.0.0.1:8000/v1 --api-key-var LOCAL_KEY \
    --num-examples 20 --rollouts-per-example 1 \
    --max-concurrent 1 --max-tokens 2048 --temperature 0.3 \
    --save-results --output-dir "$odir" \
    --env-args '{"judge_model": "openai/gpt-5.4-nano", "judge_api_key_var": "OPENROUTER_API_KEY"}' \
    > "$odir/vf-eval.log" 2>&1
  echo "[$tag] done at $(date +%H:%M:%S)"
  pkill -f serve_mcp_hypersteer 2>/dev/null; sleep 1
}

# 12 cids × 2 factors (0.3 + 0.5) = 24 cells total
for cid in 0 1 2 3 4 5 6 7 8 9 10 11; do
  run_one "v22_cid${cid}_f0p3" "$cid" 0.3
  run_one "v22_cid${cid}_f0p5" "$cid" 0.5
done

echo "============== SCORE v22 =============="
uv run --no-sync python axbench/outputs/_eval_tools/score_mcp_tox.py "$OUT_ROOT"
echo COMPLETE > /tmp/v22_eval_done.flag
echo COMPLETE > /tmp/box_b_chain_done.flag
echo "[$(date +%H:%M:%S)] Box B chain done. Output: $OUT_ROOT"
