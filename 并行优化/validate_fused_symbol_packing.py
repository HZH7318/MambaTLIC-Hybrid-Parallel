"""Verify that fused 14-stream symbol packing is codec-identical to P2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from benchmark_codec_time_method import byte_count, load_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="landsat8", choices=("landsat8", "rs"))
    parser.add_argument("--lambda-value", type=int, default=16384)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument(
        "--reference-method", default="hybrid_searchsorted_14stream_4w"
    )
    parser.add_argument(
        "--candidate-method", default="hybrid_searchsorted_fusedpack_4w"
    )
    return parser.parse_args()


def model_args(method, args):
    return SimpleNamespace(
        method=method,
        dataset=args.dataset,
        lambda_value=args.lambda_value,
        device=args.device,
    )


@torch.inference_mode()
def encode_variant(method, args, paths):
    model, _ = load_model(model_args(method, args))
    rows = []
    for path in paths:
        image = (
            torch.from_numpy(np.load(path).astype(np.float32))
            .unsqueeze(0)
            .to(args.device)
            / 65535.0
        )
        encoded = model.compress(image, return_debug=True)
        decoded = model.decompress(
            encoded["strings"], encoded["shape"], return_debug=True
        )
        torch.cuda.synchronize(torch.device(args.device))
        rows.append(
            {
                "strings": encoded["strings"],
                "shape": tuple(encoded["shape"]),
                "bytes": byte_count(encoded["strings"]),
                "encoder_y_hat": encoded["debug_y_hat"].cpu(),
                "decoder_y_hat": decoded["debug_y_hat"].cpu(),
            }
        )
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    args = parse_args()
    root = (
        Path("/data/huzhenghao/ml/test/testdata")
        / args.dataset
        / "baseall"
        / "val"
    )
    paths = sorted(root.glob("*.npy"))[: args.num_images]
    if len(paths) != args.num_images:
        raise RuntimeError(f"Need {args.num_images} images, found {len(paths)}")

    reference = encode_variant(args.reference_method, args, paths)
    fused = encode_variant(args.candidate_method, args, paths)
    details = []
    for path, left, right in zip(paths, reference, fused):
        item = {
            "image": path.name,
            "stream_bytes": left["bytes"],
            "byte_identical": left["strings"] == right["strings"],
            "shape_identical": left["shape"] == right["shape"],
            "encoder_latent_identical": torch.equal(
                left["encoder_y_hat"], right["encoder_y_hat"]
            ),
            "reference_roundtrip_exact": torch.equal(
                left["encoder_y_hat"], left["decoder_y_hat"]
            ),
            "fused_roundtrip_exact": torch.equal(
                right["encoder_y_hat"], right["decoder_y_hat"]
            ),
        }
        details.append(item)
    passed = all(
        all(value for key, value in item.items() if key.endswith("identical") or key.endswith("exact"))
        for item in details
    )
    result = {
        "passed": passed,
        "images": args.num_images,
        "reference_method": args.reference_method,
        "candidate_method": args.candidate_method,
        "details": details,
    }
    print("RESULT_JSON=" + json.dumps(result), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
