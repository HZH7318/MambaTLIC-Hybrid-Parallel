#!/usr/bin/env bash
set -euo pipefail

ML_ROOT="${ML_ROOT:-/data/huzhenghao/ml}"
ROOT="$ML_ROOT/新方法"
PYTHON="$ROOT/.venvs/chwdta/bin/python"
RUNNER="$ROOT/scripts/run_landsat8_chwdta_glic_100.py"

if [[ ! -x "$PYTHON" || ! -f "$RUNNER" ]]; then
    echo "Environment or runner missing; complete pull/setup/verify first." >&2
    exit 1
fi

for session in second_glic_then_chwdta_low second_chwdta_high; do
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "tmux session already exists: $session" >&2
        exit 1
    fi
done

# GPU 0 first finishes any GLIC lambda that has not reached 100, then handles
# the three low-rate ChWDTA continuations.
gpu0_command=(
    "$PYTHON" -u "$RUNNER"
    --gpu 0
    --phases glic_to_100
    --lambdas 1024 2048 4096 8192 16384 32768 65536
    --pipeline-name second_gpu0_glic
)
gpu0_followup=(
    "$PYTHON" -u "$RUNNER"
    --gpu 0
    --phases chwdta_to_100
    --lambdas 1024 2048 4096
    --pipeline-name second_gpu0_chwdta_low
)

# GPU 1 immediately handles the four high-rate ChWDTA continuations.
gpu1_command=(
    "$PYTHON" -u "$RUNNER"
    --gpu 1
    --phases chwdta_to_100
    --lambdas 8192 16384 32768 65536
    --pipeline-name second_gpu1_chwdta_high
)

printf -v gpu0_first '%q ' "${gpu0_command[@]}"
printf -v gpu0_second '%q ' "${gpu0_followup[@]}"
printf -v gpu1_only '%q ' "${gpu1_command[@]}"

tmux new-session -d -s second_glic_then_chwdta_low "$gpu0_first && $gpu0_second"
tmux new-session -d -s second_chwdta_high "$gpu1_only"

echo "Started GPU 0: second_glic_then_chwdta_low"
echo "Started GPU 1: second_chwdta_high"
echo "Use 'tmux attach -t SESSION' to inspect either job."
