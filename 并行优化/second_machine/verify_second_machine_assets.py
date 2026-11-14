#!/usr/bin/env python3
"""Verify reduced datasets and transferred comparison checkpoints."""

from pathlib import Path
import sys

import torch


ML_ROOT = Path("/data/huzhenghao/ml")
ROOT = ML_ROOT / "新方法"
EXPECTED = {
    "Landsat8": {"train": 1000, "val": 200, "test": 200},
    "RS": {"train": 1000, "val": 200, "test": 200},
    "FMoW": {"train": 1000, "val": 300, "test": 200},
}
LAMBDAS = (1024, 2048, 4096, 8192, 16384, 32768, 65536)


def verify_manifest(dataset: str, split: str, expected: int) -> None:
    manifest = ROOT / "data_manifests" / dataset / f"{split}.txt"
    if not manifest.is_file():
        print(f"SKIP {dataset}/{split}: manifest missing")
        return
    paths = [Path(line.strip()) for line in manifest.read_text().splitlines() if line.strip()]
    missing = [path for path in paths if not path.is_file()]
    if len(paths) != expected or missing:
        raise RuntimeError(
            f"{dataset}/{split}: expected {expected}, listed {len(paths)}, missing {len(missing)}"
        )
    print(f"OK dataset {dataset}/{split}: {len(paths)} files")


def verify_checkpoint(method: str, lmbda: int, kind: str) -> None:
    if method == "GLIC":
        folder = ROOT / "experiments" / method / "Landsat8" / f"lambda_{float(lmbda)}"
    else:
        folder = ROOT / "experiments" / method / "Landsat8" / str(float(lmbda))
    path = folder / f"checkpoint_{kind}.pth.tar"
    if not path.is_file():
        print(f"MISSING checkpoint {method} lambda={lmbda} {kind}: {path}")
        return
    checkpoint = torch.load(path, map_location="cpu")
    print(
        f"OK checkpoint {method} lambda={lmbda} {kind}: "
        f"epoch={checkpoint.get('epoch')} step={checkpoint.get('global_step')} "
        f"loss={checkpoint.get('loss')} complete={checkpoint.get('training_complete', False)}"
    )
    del checkpoint


def main() -> None:
    for dataset, splits in EXPECTED.items():
        for split, expected in splits.items():
            verify_manifest(dataset, split, expected)
    for method in ("GLIC", "ChWDTA"):
        for lmbda in LAMBDAS:
            for kind in ("latest", "best"):
                verify_checkpoint(method, lmbda, kind)
    vendor = ML_ROOT / "h" / "并行优化" / "vendor" / "rwk_compressai126" / "compressai"
    if not vendor.is_dir():
        raise RuntimeError(f"Missing GLIC CompressAI vendor snapshot: {vendor}")
    print("Asset verification finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"VERIFY FAILED: {error}", file=sys.stderr)
        raise
