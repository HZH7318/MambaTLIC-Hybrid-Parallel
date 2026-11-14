"""Re-evaluate all available codecs on the current three validation sets.

This script deliberately does not read RD values from the historical plotting
notebooks.  It reuses the model adapters and checkpoints under ``/ml/test``,
but reads the exact current validation manifests under ``/ml/新方法``.

Protocol:
  * BPP is estimated from forward likelihoods (not real bitstream bytes).
  * PSNR and MS-SSIM are computed after clipping reconstruction to [0, 1].
  * Landsat8, legacy RS and FMoW weights expect 65535, 65535 and 4096.
  * Results are checkpoint-resumable at one dataset/method/lambda point.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pytorch_msssim import ms_ssim


ROOT = Path(__file__).resolve().parent
ML_ROOT = ROOT.parent.parent
TEST_ROOT = ML_ROOT / "test"
MANIFEST_ROOT = ML_ROOT / "新方法" / "data_manifests"
WEIGHT_ROOT = TEST_ROOT / "pth"
OUTPUT_ROOT = ROOT / "results" / "val_retest"
SUMMARY_CSV = ROOT / "results" / "data" / "three_datasets_val_rd.csv"

DATASETS = {
    "landsat8": {
        "manifest": MANIFEST_ROOT / "Landsat8" / "val.txt",
        "weight_name": "landsat8",
        "channels": 7,
        "normalization": 65535.0,
    },
    "rs": {
        "manifest": MANIFEST_ROOT / "RS" / "val.txt",
        "weight_name": "rs",
        "channels": 10,
        # All checkpoints evaluated here (including the current Hybrid RS)
        # were trained with this exact value. A future /32767 retraining must
        # be reported as a separate protocol and must not reuse these weights.
        "normalization": 65535.0,
    },
    "fmow8": {
        "manifest": MANIFEST_ROOT / "FMoW" / "val.txt",
        "weight_name": "fmow8",
        "channels": 8,
        "normalization": 4096.0,
    },
}

LEGACY_METHODS = ("elic", "rwk", "hl", "mlic", "ftic", "hpcm", "our")
DISPLAY_NAMES = {
    "elic": "ELIC",
    "rwk": "RWK",
    "hl": "HL",
    "mlic": "MLIC",
    "ftic": "FTIC",
    "hpcm": "HPCM",
    "our": "Our-Original",
    "hybrid_parallel": "Hybrid-Parallel",
}

HYBRID_CHECKPOINTS = {
    "landsat8": {
        1024: ROOT / "experiments/train/20261018_2331_1024/bestmodel.pth",
        2048: ROOT / "experiments/train/20261019_0741_2048/bestmodel.pth",
        4096: ROOT / "experiments/train/20261019_1638_4096/bestmodel.pth",
        8192: ROOT / "experiments/train/20261019_2052_8192/bestmodel.pth",
        16384: ROOT / "experiments/train/20261019_0334_16384/bestmodel.pth",
        32768: ROOT / "experiments/train/20261019_1145_32768/bestmodel.pth",
        65536: ROOT / "experiments/train/20261020_0055_65536/bestmodel.pth",
    },
    "rs": {
        32768: ROOT / "experiments/RS/train/20261020_1141_32768/bestmodel.pth",
        65536: ROOT / "experiments/RS/train/20261020_2222_65536/bestmodel.pth",
        131071: ROOT / "experiments/RS/train/20261021_0904_131071/bestmodel.pth",
        262144: ROOT / "experiments/RS/train/20261021_1946_262144/bestmodel.pth",
        393216: ROOT / "experiments/RS/train/20261022_0629_393216/bestmodel.pth",
        524288: ROOT / "experiments/RS/train/20261022_1711_524288/bestmodel.pth",
    },
    "fmow8": {
        value: ROOT / (
            "experiments/FMoW_4096_representative_1000_300_300/"
            f"train/lambda_{value}/bestmodel.pth"
        )
        for value in (1024, 2048, 4096, 8192, 16384, 32768, 65536)
    },
}

FIELDNAMES = (
    "dataset", "method", "lambda", "images", "normalization",
    "bpp", "psnr", "ms_ssim_db", "bpp_std", "psnr_std",
    "ms_ssim_db_std", "checkpoint_epoch", "checkpoint", "protocol",
    "elapsed_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS)
    )
    parser.add_argument(
        "--methods", nargs="+", choices=(*LEGACY_METHODS, "hybrid_parallel"),
        default=[*LEGACY_METHODS, "hybrid_parallel"],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lambdas", nargs="+", type=int)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    return parser.parse_args()


def checkpoint_lambda(path: Path) -> int | None:
    try:
        return int(path.name.split(".", 1)[0])
    except ValueError:
        return None


def discover_legacy_checkpoints(method: str, dataset: str) -> dict[int, Path]:
    folder = WEIGHT_ROOT / method / DATASETS[dataset]["weight_name"]
    result = {}
    if not folder.is_dir():
        return result
    for path in folder.iterdir():
        if not path.is_file():
            continue
        value = checkpoint_lambda(path)
        if value is not None:
            result[value] = path
    return dict(sorted(result.items()))


def read_manifest(path: Path, max_images: int | None) -> list[Path]:
    paths = [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if max_images is not None:
        paths = paths[:max_images]
    missing = [str(item) for item in paths if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Manifest {path} has missing files: {missing[:3]}")
    return paths


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


def load_hybrid(checkpoint_path: Path, channels: int, device: str):
    project_root = ROOT.parent
    model_path = ROOT / "model"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(model_path) not in sys.path:
        sys.path.insert(0, str(model_path))
    from mambamodel_hybrid_parallel import HNet

    model = HNet(192, 320, channels, entropy_workers=1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    incompatible = model.load_state_dict(remap_legacy_entropy_keys(state), strict=False)
    missing = [
        key for key in incompatible.missing_keys
        if not key.endswith(("._offset", "._quantized_cdf", "._cdf_length"))
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Hybrid checkpoint mismatch: missing={missing[:10]}, "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )
    return model.eval().to(device), checkpoint.get("epoch")


def load_legacy(method: str, dataset: str, value: int, channels: int, device: str):
    if str(TEST_ROOT) in sys.path:
        sys.path.remove(str(TEST_ROOT))
    sys.path.insert(0, str(TEST_ROOT))
    from testmodel import getLoad, getNetByMethod

    if method == "rwk":
        # The original nl3c environment has a valid prebuilt torch-1.12 CUDA
        # extension.  Rebuilding it now would incorrectly use the machine's
        # newer CUDA-13 toolkit, which is incompatible with torch-1.12's
        # generated C++14 build. Load the proven binary directly.
        extension = Path(
            "/data/huzhenghao/.cache/torch_extensions/py39_cu112/"
            "biwkv4/biwkv4.so"
        )
        if not extension.is_file():
            raise FileNotFoundError(f"RWK extension missing: {extension}")
        torch.ops.load_library(str(extension))
        from model.rwk.models import lalic

        lalic.load_biwkv4 = lambda: None
        model = lalic.LALIC(channel=channels)
    else:
        model = getNetByMethod(method, 192, 320, channels)
    model = getLoad(model, method, DATASETS[dataset]["weight_name"], value)
    if model is None:
        raise FileNotFoundError(f"Could not load {method}/{dataset}/{value}")
    return model.eval().to(device), None


def calculate_metrics(output: dict, target: torch.Tensor) -> dict[str, float]:
    pixels = target.shape[0] * target.shape[-2] * target.shape[-1]
    if "ishl" in output:
        bpp = output["bpp"]
    else:
        likelihoods = output.get("likelihoods", output.get("img_likelihoods"))
        if likelihoods is None:
            raise KeyError("Output has neither likelihoods nor img_likelihoods")
        bpp = sum(
            torch.log(item).sum() / (-math.log(2) * pixels)
            for item in likelihoods.values()
        )
    reconstruction = output["x_hat"].clamp(0.0, 1.0)
    mse = torch.mean((reconstruction - target) ** 2)
    similarity = ms_ssim(reconstruction, target, data_range=1.0)
    return {
        "bpp": float(bpp.item()),
        "psnr": float((-10.0 * torch.log10(mse)).item()),
        "ms_ssim_db": float((-10.0 * torch.log10(1.0 - similarity)).item()),
    }


def read_existing(summary_csv: Path) -> list[dict[str, str]]:
    if not summary_csv.is_file():
        return []
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_summary(rows: list[dict], summary_csv: Path) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_csv.with_suffix(".tmp")
    rows = sorted(rows, key=lambda row: (
        row["dataset"], row["method"], int(row["lambda"])
    ))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(summary_csv)


@torch.inference_mode()
def evaluate_point(
    model, paths: list[Path], normalization: float, device: str,
    progress_every: int, label: str,
) -> list[dict[str, float | str]]:
    rows = []
    for index, path in enumerate(paths, 1):
        array = np.load(path).astype(np.float32)
        target = torch.from_numpy(array).unsqueeze(0).to(device)
        target = target / normalization
        metrics = calculate_metrics(model(target), target)
        rows.append({"file": str(path), **metrics})
        if index == 1 or index % progress_every == 0 or index == len(paths):
            print(
                f"{label} {index}/{len(paths)} "
                f"BPP={metrics['bpp']:.5f} PSNR={metrics['psnr']:.3f}",
                flush=True,
            )
    return rows


def main() -> None:
    args = parse_args()
    existing = read_existing(args.summary_csv)
    complete = {
        (row["dataset"], row["method"], int(row["lambda"])) for row in existing
    }
    failures = []
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        paths = read_manifest(spec["manifest"], args.max_images)
        for method in args.methods:
            checkpoints = (
                HYBRID_CHECKPOINTS[dataset]
                if method == "hybrid_parallel"
                else discover_legacy_checkpoints(method, dataset)
            )
            if args.lambdas:
                checkpoints = {
                    value: path for value, path in checkpoints.items()
                    if value in args.lambdas
                }
            for value, checkpoint_path in checkpoints.items():
                key = (dataset, DISPLAY_NAMES[method], value)
                if key in complete and not args.force and args.max_images is None:
                    print(f"SKIP complete: {key}", flush=True)
                    continue
                if not checkpoint_path.is_file():
                    failures.append({"key": key, "error": f"missing {checkpoint_path}"})
                    continue
                model = None
                started = time.monotonic()
                try:
                    if method == "hybrid_parallel":
                        model, epoch = load_hybrid(
                            checkpoint_path, spec["channels"], args.device
                        )
                    else:
                        model, epoch = load_legacy(
                            method, dataset, value, spec["channels"], args.device
                        )
                    label = f"{dataset}/{DISPLAY_NAMES[method]}/lambda={value}"
                    per_image = evaluate_point(
                        model, paths, spec["normalization"], args.device,
                        args.progress_every, label,
                    )
                    elapsed = time.monotonic() - started
                    result = {
                        "dataset": dataset,
                        "method": DISPLAY_NAMES[method],
                        "lambda": value,
                        "images": len(per_image),
                        "normalization": spec["normalization"],
                        **{
                            metric: float(np.mean([row[metric] for row in per_image]))
                            for metric in ("bpp", "psnr", "ms_ssim_db")
                        },
                        **{
                            f"{metric}_std": float(np.std([row[metric] for row in per_image]))
                            for metric in ("bpp", "psnr", "ms_ssim_db")
                        },
                        "checkpoint_epoch": "" if epoch is None else epoch,
                        "checkpoint": str(checkpoint_path),
                        "protocol": "val_forward_likelihood_bpp",
                        "elapsed_seconds": elapsed,
                    }
                    existing = [
                        row for row in existing
                        if (row["dataset"], row["method"], int(row["lambda"]))
                        != (dataset, DISPLAY_NAMES[method], value)
                    ]
                    existing.append(result)
                    save_summary(existing, args.summary_csv)
                    complete.add(key)
                    print(f"DONE {label}: {json.dumps(result, ensure_ascii=False)}", flush=True)
                except Exception as error:
                    failures.append({"key": key, "error": repr(error)})
                    print(f"FAILED {key}: {error!r}", file=sys.stderr, flush=True)
                finally:
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Summary: {args.summary_csv}")
    print(f"Failures: {len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
