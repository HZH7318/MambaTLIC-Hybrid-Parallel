#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
benchmark="$script_dir/benchmark_codec_time_method.py"
base_python="/data/huzhenghao/anaconda3/bin/python"
mamba_python="/data/huzhenghao/anaconda3/envs/mambah/bin/python"
rwk_python="/data/huzhenghao/anaconda3/envs/nl3c/bin/python"

method=""
previous=""
for argument in "$@"; do
    if [[ "$previous" == "--method" ]]; then
        method="$argument"
        break
    fi
    previous="$argument"
done
if [[ -z "$method" ]]; then
    echo "Missing required --method argument" >&2
    exit 2
fi

if [[ "$method" == hybrid_* || "$method" == "mamba_serial" ]]; then
    exec "$mamba_python" "$benchmark" "$@"
elif [[ "$method" == "rwk" ]]; then
    exec "$rwk_python" "$benchmark" "$@"
else
    exec "$base_python" "$benchmark" "$@"
fi
