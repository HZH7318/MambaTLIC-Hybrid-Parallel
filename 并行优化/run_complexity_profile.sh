#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profiler="$script_dir/profile_complexity_method.py"
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

if [[ "$method" == "hybrid_parallel" || "$method" == "mamba_serial" ||
      "$method" == "our_original" || "$method" == "our_native_1t" ]]; then
    exec "$mamba_python" "$profiler" "$@"
elif [[ "$method" == "rwk" ]]; then
    exec "$rwk_python" "$profiler" "$@"
else
    exec "$base_python" "$profiler" "$@"
fi
