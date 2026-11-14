#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 SOURCE_SSH [landsat8|all|none]" >&2
    echo "Example: $0 huzhenghao@server2 landsat8" >&2
    exit 2
fi

SOURCE_SSH="$1"
DATA_SCOPE="${2:-landsat8}"
ML_ROOT="${ML_ROOT:-/data/huzhenghao/ml}"
H_ROOT="$ML_ROOT/h"
NEW_METHOD_ROOT="$ML_ROOT/新方法"
GIT_URL="https://github.com/HZH7318/MambaTLIC-Hybrid-Parallel.git"
LAMBDAS=(1024 2048 4096 8192 16384 32768 65536)

case "$DATA_SCOPE" in
    landsat8|all|none) ;;
    *) echo "Invalid data scope: $DATA_SCOPE" >&2; exit 2 ;;
esac

mkdir -p "$ML_ROOT"

if [[ ! -d "$H_ROOT/.git" ]]; then
    git clone "$GIT_URL" "$H_ROOT"
else
    git -C "$H_ROOT" fetch origin main
    if [[ "${UPDATE_H:-0}" == "1" ]]; then
        git -C "$H_ROOT" pull --ff-only origin main
    else
        echo "Existing h repository preserved; fetched origin/main without changing its worktree."
    fi
fi

mkdir -p "$NEW_METHOD_ROOT"
rsync -a -s --info=progress2 \
    --exclude '.venvs/' \
    --exclude 'experiments/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    "$SOURCE_SSH:$ML_ROOT/新方法/" "$NEW_METHOD_ROOT/"

# This audited CompressAI snapshot is intentionally ignored by the h Git repo,
# but GLIC imports it through PYTHONPATH.
mkdir -p "$H_ROOT/并行优化/vendor/rwk_compressai126"
rsync -a -s --info=progress2 \
    --exclude '__pycache__/' --exclude '*.pyc' \
    "$SOURCE_SSH:$ML_ROOT/h/并行优化/vendor/rwk_compressai126/" \
    "$H_ROOT/并行优化/vendor/rwk_compressai126/"

sync_dataset() {
    local dataset="$1"
    local split manifest file_list
    for split in train val test; do
        manifest="$NEW_METHOD_ROOT/data_manifests/$dataset/$split.txt"
        if [[ ! -f "$manifest" ]]; then
            echo "Missing manifest after code sync: $manifest" >&2
            exit 1
        fi
        file_list="$(mktemp)"
        sed 's#^/##' "$manifest" > "$file_list"
        # Manifests may point at the reduced-set symlinks used on server2.
        # Dereference them so the target receives self-contained .npy files.
        rsync -aL -s --partial --info=progress2 --files-from="$file_list" \
            "$SOURCE_SSH:/" / 
        rm -f "$file_list"
    done
}

if [[ "$DATA_SCOPE" == "landsat8" ]]; then
    sync_dataset Landsat8
elif [[ "$DATA_SCOPE" == "all" ]]; then
    sync_dataset Landsat8
    sync_dataset RS
    sync_dataset FMoW
fi

# Copy only resume/final-selection checkpoints. Historical numbered checkpoints
# account for most of the 214 GB source workspace and are not needed.
for lmbda in "${LAMBDAS[@]}"; do
    chwdta_dir="$NEW_METHOD_ROOT/experiments/ChWDTA/Landsat8/${lmbda}.0"
    glic_dir="$NEW_METHOD_ROOT/experiments/GLIC/Landsat8/lambda_${lmbda}.0"
    mkdir -p "$chwdta_dir" "$glic_dir"
    for checkpoint_name in checkpoint_latest.pth.tar checkpoint_best.pth.tar; do
        rsync -a -s --partial --info=progress2 \
            "$SOURCE_SSH:$ML_ROOT/新方法/experiments/ChWDTA/Landsat8/${lmbda}.0/$checkpoint_name" \
            "$chwdta_dir/" || true
        rsync -a -s --partial --info=progress2 \
            "$SOURCE_SSH:$ML_ROOT/新方法/experiments/GLIC/Landsat8/lambda_${lmbda}.0/$checkpoint_name" \
            "$glic_dir/" || true
    done
done

mkdir -p "$NEW_METHOD_ROOT/experiments/pipelines"
rsync -a -s --info=progress2 \
    "$SOURCE_SSH:$ML_ROOT/新方法/experiments/pipelines/" \
    "$NEW_METHOD_ROOT/experiments/pipelines/" || true

echo "Transfer finished. Run setup_second_machine_env.sh, then verify_second_machine_assets.py."
