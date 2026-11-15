"""Unified real-codec benchmark for the S0/S1/S2 structural ablation."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pytorch_msssim import ms_ssim


ROOT = Path(__file__).resolve().parents[2]
OPT_ROOT = ROOT / "并行优化"
sys.path[:0] = [
    str(ROOT),
    str(OPT_ROOT),
    str(OPT_ROOT / "model"),
    str(Path(__file__).resolve().parent),
]

from mambamodel_hybrid_parallel import HNet as S2Net  # noqa: E402
from model_serial_grouped import S0Net, S1Net  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("s0", "s1", "s2"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-images", type=int, default=5)
    parser.add_argument("--num-images", type=int, default=30)
    return parser.parse_args()


def byte_count(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, (list, tuple)):
        return sum(byte_count(item) for item in value)
    if isinstance(value, dict):
        return sum(byte_count(item) for item in value.values())
    return 0


def timed(device, function):
    torch.cuda.synchronize(torch.device(device))
    started = time.perf_counter()
    output = function()
    torch.cuda.synchronize(torch.device(device))
    return output, time.perf_counter() - started


def normalized_state(checkpoint):
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    remapped = {}
    pattern = re.compile(r"\._(matrix|bias|factor)(\d+)(?=\.|$)")
    plural = {"matrix": "matrices", "bias": "biases", "factor": "factors"}
    for key, value in state.items():
        match = pattern.search(key)
        if match:
            key = (
                key[: match.start()]
                + f".{plural[match.group(1)]}.{match.group(2)}"
                + key[match.end() :]
            )
        remapped[key] = value
    return remapped


def load_model(args):
    if args.variant == "s0":
        wrapper = S0Net(192, 320, 7)
    elif args.variant == "s1":
        wrapper = S1Net(192, 320, 7)
    else:
        wrapper = S2Net(
            192,
            320,
            7,
            entropy_workers=4,
            use_native_entropy=True,
            use_searchsorted_indexes=True,
            defer_native_encode=True,
        )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    wrapper.load_state_dict(normalized_state(checkpoint), strict=True)
    wrapper.eval().to(args.device)
    wrapper.net.update(force=True)
    return wrapper.net


def summary(values):
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "p90": float(np.percentile(values, 90)),
    }


@torch.inference_mode()
def main():
    args = parse_args()
    model = load_model(args)
    paths = sorted(
        Path("/data/huzhenghao/ml/test/testdata/landsat8/baseall/val").glob("*.npy")
    )
    required = args.warmup_images + args.num_images
    if len(paths) < required:
        raise RuntimeError(f"Need {required} images, found {len(paths)}")
    rows = []
    for index, path in enumerate(paths[:required]):
        image = (
            torch.from_numpy(np.load(path).astype(np.float32))
            .unsqueeze(0)
            .to(args.device)
            / 65535.0
        )
        encoded, enc_s = timed(
            args.device, lambda: model.compress(image, return_debug=True)
        )
        decoded, dec_s = timed(
            args.device,
            lambda: model.decompress(
                encoded["strings"], encoded["shape"], return_debug=True
            ),
        )
        if index < args.warmup_images:
            continue
        reconstruction = decoded["x_hat"].clamp(0, 1)
        mse = torch.mean((reconstruction - image) ** 2)
        similarity = ms_ssim(reconstruction, image, data_range=1.0)
        rows.append(
            {
                "enc_s": enc_s,
                "dec_s": dec_s,
                "bpp": byte_count(encoded["strings"])
                * 8.0
                / (image.shape[-2] * image.shape[-1]),
                "psnr": float(-10 * torch.log10(mse)),
                "ms_ssim": float(similarity),
                "latent_exact": torch.equal(
                    encoded["debug_y_hat"], decoded["debug_y_hat"]
                ),
                "streams": sum(len(group) for group in encoded["strings"][0])
                + 1,
            }
        )
        print(f"{args.variant} {len(rows)}/{args.num_images}", flush=True)
    result = {
        "variant": args.variant,
        "checkpoint": str(args.checkpoint),
        "images": args.num_images,
        "compress_s": summary([row["enc_s"] for row in rows]),
        "decompress_s": summary([row["dec_s"] for row in rows]),
        "actual_bitstream_bpp": statistics.fmean(row["bpp"] for row in rows),
        "psnr": statistics.fmean(row["psnr"] for row in rows),
        "ms_ssim": statistics.fmean(row["ms_ssim"] for row in rows),
        "ms_ssim_db": statistics.fmean(
            -10.0 * np.log10(max(1.0 - row["ms_ssim"], 1e-12)) for row in rows
        ),
        "parameters_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "streams": sorted({row["streams"] for row in rows}),
        "latent_exact": all(row["latent_exact"] for row in rows),
        "bpp_protocol": "actual bitstream bytes / image pixels",
    }
    print("RESULT_JSON=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
