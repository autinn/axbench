#!/bin/bash
# Factor sweep against trained HyperSteer (v1).
# For each factor: kill+restart serve, wait for healthz, run vf-eval n=20, save abbreviated summary.
set -uo pipefail
cd ~/mcp-protect
export PATH=$HOME/.local/bin:$PATH
set -a; source ~/mcp-protect/.env; set +a

DUMP="axbench/outputs/mcp_hsteer_9b_v1"
CONCEPT_ID="${CONCEPT_ID:-147}"
N="${N:-20}"
SUMMARY=~/eval_out/v1_sweep_summary.txt
mkdir -p ~/eval_out
echo "factor concept_id N attack_resistance attack_detected num_turns errors" > "$SUMMARY"

for FACTOR in 0.0 0.5 1.0 1.5 2.0; do
    TAG="v1_c${CONCEPT_ID}_f${FACTOR}_n${N}"
    OUT=~/eval_out/$TAG
    rm -rf "$OUT"
    echo
    echo "===== [$(date +%H:%M:%S)] FACTOR=$FACTOR ====="

    # restart serve
    pkill -9 -f serve_mcp_hypersteer.py 2>/dev/null || true
    sleep 3
    cd ~/mcp-protect/axbench
    HYPERSTEER_CONCEPT_ID=$CONCEPT_ID HYPERSTEER_FACTOR=$FACTOR \
        OPENAI_API_KEY=$OPENROUTER_API_KEY OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
        nohup uv run python axbench/mcp-protect/serve_mcp_hypersteer.py \
        --dump-dir "$DUMP" --port 8000 > /tmp/serve_$FACTOR.log 2>&1 &
    SERVE_PID=$!
    cd ~/mcp-protect

    # wait for healthz (max 5 min)
    for i in $(seq 1 60); do
        if curl -sf http://127.0.0.1:8000/healthz >/dev/null 2>&1; then break; fi
        sleep 5
    done
    if ! curl -sf http://127.0.0.1:8000/healthz >/dev/null 2>&1; then
        echo "$FACTOR $CONCEPT_ID $N - - - serve_failed" >> "$SUMMARY"
        echo "[FATAL] serve never came up at factor=$FACTOR; tail of serve log:"
        tail -40 /tmp/serve_$FACTOR.log
        continue
    fi

    # run vf-eval
    cd ~/mcp-protect/prime-envs
    OPENAI_API_KEY=dummy OPENROUTER_API_KEY=$OPENROUTER_API_KEY uv run vf-eval mcp_tox \
        -m hypersteer-local \
        --api-base-url http://127.0.0.1:8000/v1 \
        --api-key-var OPENAI_API_KEY \
        --api-client-type openai_chat_completions \
        --num-examples $N --rollouts-per-example 1 --max-concurrent 4 \
        --max-tokens 512 --temperature 0.0 \
        --extra-env-kwargs '{"judge_model":"openai/gpt-5.4-nano","judge_api_key_var":"OPENROUTER_API_KEY","judge_base_url":"https://openrouter.ai/api/v1"}' \
        --output-dir "$OUT" --save-results --abbreviated-summary 2>&1 | tee /tmp/eval_$FACTOR.log
    cd ~/mcp-protect

    # parse summary
    AR=$(grep -m1 "^attack_resistance:" /tmp/eval_$FACTOR.log | grep -oE "avg - [0-9.]+" | awk '{print $3}')
    AD=$(grep -m1 "^attack_detected:" /tmp/eval_$FACTOR.log | grep -oE "avg - [0-9.]+" | awk '{print $3}')
    NT=$(grep -m1 "^num_turns:" /tmp/eval_$FACTOR.log | grep -oE "avg - [0-9.]+" | awk '{print $3}')
    ERR=$(grep -m1 "^errors:" /tmp/eval_$FACTOR.log | grep -oE "avg - [0-9.]+" | awk '{print $3}')
    echo "$FACTOR $CONCEPT_ID $N ${AR:-?} ${AD:-?} ${NT:-?} ${ERR:-?}" >> "$SUMMARY"
done

# cleanup serve
pkill -9 -f serve_mcp_hypersteer.py 2>/dev/null || true
echo
echo "===== SWEEP COMPLETE ====="
cat "$SUMMARY"
