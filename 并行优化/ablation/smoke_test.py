"""One-step training and real-codec smoke test for S0/S1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
OPT_ROOT = ROOT / "并行优化"
sys.path[:0] = [
    str(ROOT),
    str(OPT_ROOT),
    str(OPT_ROOT / "model"),
    str(Path(__file__).resolve().parent),
]

from model_serial_grouped import RateDistortion, S0Net, S1Net  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variant", choices=("s0", "s1", "both"), default="both")
    parser.add_argument(
        "--s0-checkpoint",
        type=Path,
        default=Path("/data/huzhenghao/ml/test/pth/our/landsat8/16384.pth"),
    )
    return parser.parse_args()


def byte_count(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, (list, tuple)):
        return sum(byte_count(item) for item in value)
    if isinstance(value, dict):
        return sum(byte_count(item) for item in value.values())
    return 0


def run_variant(name, cls, image, args):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    model = cls(192, 320, 7)
    if name == "s0":
        checkpoint = torch.load(args.s0_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
    model.train().to(args.device)
    criterion = RateDistortion(16384)
    output = model(image)
    losses = criterion(output, image)
    losses["loss"].backward()
    train_loss = float(losses["loss"].detach())
    del output, losses
    model.zero_grad(set_to_none=True)

    model.eval()
    model.net.update(force=True)
    with torch.inference_mode():
        encoded = model.net.compress(image, return_debug=True)
        decoded = model.net.decompress(
            encoded["strings"], encoded["shape"], return_debug=True
        )
    torch.cuda.synchronize(torch.device(args.device))
    result = {
        "variant": name,
        "parameters_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "train_loss": train_loss,
        "streams": sum(len(group) for group in encoded["strings"][0]) + 1,
        "bytes": byte_count(encoded["strings"]),
        "latent_exact": torch.equal(
            encoded["debug_y_hat"], decoded["debug_y_hat"]
        ),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(args.device)
        / (1024**3),
    }
    del model, encoded, decoded
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    path = sorted(
        Path("/data/huzhenghao/ml/test/testdata/landsat8/baseall/val").glob("*.npy")
    )[0]
    image = (
        torch.from_numpy(np.load(path).astype(np.float32))
        .unsqueeze(0)
        .to(args.device)
        / 65535.0
    )
    variants = {"s0": S0Net, "s1": S1Net}
    selected = variants if args.variant == "both" else {args.variant: variants[args.variant]}
    results = [run_variant(name, cls, image, args) for name, cls in selected.items()]
    print("RESULT_JSON=" + json.dumps(results), flush=True)
    if not all(item["latent_exact"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
