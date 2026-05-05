#!/bin/bash
set -euo pipefail

cd ~/mcp-protect/axbench
export PATH=$HOME/.local/bin:$PATH

set -a
source ~/mcp-protect/.env
set +a

export OPENAI_API_KEY="${OPENROUTER_API_KEY}"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
# Reduce CUDA fragmentation for the two-Gemma-9b setup.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null || true

# Patch hardcoded gpt-4o-mini → openai/gpt-4o-mini (idempotent).
grep -q '"openai/gpt-4o-mini"' axbench/mcp-protect/mcp_hypersteer.py || \
  sed -i 's|"lm_model": "gpt-4o-mini"|"lm_model": "openai/gpt-4o-mini"|g' \
      axbench/mcp-protect/mcp_hypersteer.py

RUN_NAME="${RUN_NAME:-mcp_hsteer_9b_smoke}"
MODEL="${MODEL:-9b}"
LAYER="${LAYER:-20}"
MERGE_NEURONPEDIA="${MERGE_NEURONPEDIA:-100}"
MERGE_MCP="${MERGE_MCP:-50}"
NUM_EXAMPLES="${NUM_EXAMPLES:-8}"
N_EPOCHS="${N_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SKIP_GENERATE_FLAG=""
[ "${SKIP_GENERATE:-0}" = "1" ] && SKIP_GENERATE_FLAG="--skip-generate"
DUMP="axbench/outputs/${RUN_NAME}"

echo "[$(date +%H:%M:%S)] mcp_hypersteer model=${MODEL} layer=${LAYER} -c ${MERGE_NEURONPEDIA} -a ${MERGE_MCP} num_ex=${NUM_EXAMPLES} epochs=${N_EPOCHS} batch=${BATCH_SIZE} ${SKIP_GENERATE_FLAG} → ${DUMP}"
exec uv run python axbench/mcp-protect/mcp_hypersteer.py \
    --model "${MODEL}" --layer "${LAYER}" \
    -c "${MERGE_NEURONPEDIA}" -a "${MERGE_MCP}" \
    --num-of-examples "${NUM_EXAMPLES}" \
    --n-epochs "${N_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    ${SKIP_GENERATE_FLAG} \
    --dump-dir "${DUMP}" \
    --master-data-dir axbench/data \
    --nproc-per-node 1
