#!/usr/bin/env bash
set -euo pipefail

ML_ROOT="${ML_ROOT:-/data/huzhenghao/ml}"
NEW_METHOD_ROOT="$ML_ROOT/新方法"
BASE_PYTHON="${BASE_PYTHON:-/data/huzhenghao/anaconda3/bin/python}"
VENV_ROOT="$NEW_METHOD_ROOT/.venvs/chwdta"

if [[ ! -x "$BASE_PYTHON" ]]; then
    echo "Base Python not found: $BASE_PYTHON" >&2
    exit 1
fi

if [[ ! -x "$VENV_ROOT/bin/python" ]]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$VENV_ROOT"
fi

PYTHONPATH="$ML_ROOT/h/并行优化/vendor/rwk_compressai126${PYTHONPATH:+:$PYTHONPATH}" \
"$VENV_ROOT/bin/python" - <<'PY'
import importlib
required = {
    "torch": "2.4.1+cu121",
    "torchvision": "0.19.1+cu121",
    "compressai": "1.1.5",
    "tensorboard": "2.14.1",
    "timm": "1.0.15",
    "einops": "0.8.1",
    "pywt": "1.6.0",
    "pytorch_msssim": None,
    "numpy": "1.24.4",
}
failed = []
for name, expected in required.items():
    try:
        module = importlib.import_module(name)
        actual = getattr(module, "__version__", "unknown")
        print(f"{name}: {actual} (source server: {expected or 'installed'})")
    except Exception as error:
        failed.append((name, repr(error)))
if failed:
    print("\nMissing or broken imports:")
    for item in failed:
        print("  ", item)
    raise SystemExit(1)

import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see CUDA on this machine")
if torch.cuda.device_count() < 2:
    raise SystemExit(f"Expected two GPUs, found {torch.cuda.device_count()}")
print("CUDA devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY

echo "Environment check passed."
