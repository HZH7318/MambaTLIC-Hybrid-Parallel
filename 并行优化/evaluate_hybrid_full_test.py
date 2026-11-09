"""Evaluate hybrid checkpoints on the exact full test sets used by /ml/test.

This is an RD evaluation using model likelihoods, matching the protocol of
``/data/huzhenghao/ml/test/summary_all_results.csv``.  It is intentionally
separate from ``benchmark_real_codec.py``, which measures actual bitstreams
and complete codec latency on a smaller controlled sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch_msssim import ms_ssim


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ROOT / "model"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--lambda-value", required=True, type=int)
    parser.add_argument("--channels", required=True, type=int)
    parser.add_argument("--normalization", required=True, type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def remap_legacy_entropy_keys(state_dict: dict[str, torch.Tensor]):
    remapped = {}
    pattern = re.compile(r"\.(?:_)(matrix|bias|factor)(\d+)(?=\.|$)")
    plural = {"matrix": "matrices", "bias": "biases", "factor": "factors"}
    for key, value in state_dict.items():
        match = pattern.search(key)
        if match:
            replacement = f".{plural[match.group(1)]}.{match.group(2)}"
            key = key[: match.start()] + replacement + key[match.end() :]
        remapped[key] = value
    return remapped


def load_model(args: argparse.Namespace):
    from mambamodel_hybrid_parallel import HNet

    model = HNet(192, 320, args.channels, entropy_workers=1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    incompatible = model.load_state_dict(remap_legacy_entropy_keys(state), strict=False)
    missing = [
        key for key in incompatible.missing_keys
        if not key.endswith(("._offset", "._quantized_cdf", "._cdf_length"))
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing[:20]}, "
            f"unexpected={incompatible.unexpected_keys[:20]}"
        )
    model.eval().to(args.device)
    return model, checkpoint


def calculate_metrics(output: dict, target: torch.Tensor) -> dict[str, float]:
    _, _, height, width = target.shape
    pixels = target.shape[0] * height * width
    likelihoods = output.get("likelihoods", output.get("img_likelihoods"))
    if likelihoods is None:
        raise KeyError("Model output has neither likelihoods nor img_likelihoods")
    bpp = sum(
        torch.log(likelihood).sum() / (-math.log(2) * pixels)
        for likelihood in likelihoods.values()
    )
    reconstruction = output["x_hat"].clamp(0.0, 1.0)
    mse = torch.mean((reconstruction - target) ** 2)
    psnr = -10.0 * torch.log10(mse)
    similarity = ms_ssim(reconstruction, target, data_range=1.0)
    similarity_db = -10.0 * torch.log10(1.0 - similarity)
    return {
        "bpp": float(bpp.item()),
        "psnr": float(psnr.item()),
        "ms_ssim_db": float(similarity_db.item()),
        "mse": float(mse.item()),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    paths = sorted(args.dataset.glob("*.npy"))
    if args.max_images is not None:
        paths = paths[: args.max_images]
    if not paths:
        raise FileNotFoundError(f"No npy files found in {args.dataset}")
    model, checkpoint = load_model(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(paths, 1):
        array = np.load(path).astype(np.float32)
        target = torch.from_numpy(array).unsqueeze(0).to(args.device)
        target = target / args.normalization
        output = model(target)
        row = {"file": path.stem, **calculate_metrics(output, target)}
        rows.append(row)
        if index == 1 or index % args.progress_every == 0 or index == len(paths):
            print(
                f"{args.dataset_name} lambda={args.lambda_value}: "
                f"{index}/{len(paths)} bpp={row['bpp']:.5f} psnr={row['psnr']:.3f}",
                flush=True,
            )

    csv_path = args.output_dir / "per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "forward_likelihood_full_test_set",
        "dataset": args.dataset_name,
        "dataset_path": str(args.dataset),
        "lambda": args.lambda_value,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "normalization": args.normalization,
        "num_images": len(rows),
        "metrics": {
            key: {
                "mean": float(np.mean([row[key] for row in rows])),
                "median": float(np.median([row[key] for row in rows])),
                "std": float(np.std([row[key] for row in rows])),
            }
            for key in ("bpp", "psnr", "ms_ssim_db", "mse")
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"CSV: {csv_path}\nSummary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
