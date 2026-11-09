"""Benchmark one comparison codec and emit one JSON result line."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from pytorch_msssim import ms_ssim


OPT_ROOT = Path(__file__).resolve().parent
TEST_ROOT = Path("/data/huzhenghao/ml/test")
sys.path.insert(0, str(OPT_ROOT))
sys.path.insert(0, str(OPT_ROOT / "vendor"))
sys.path.insert(0, str(TEST_ROOT))
sys.path.insert(0, str(TEST_ROOT / "model" / "hpcm"))


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=(
            "elic", "rwk", "mlic", "hpcm", "ftic", "hl",
            "mamba_serial", "hybrid_1w", "hybrid_4w",
            "hybrid_batched_multistream", "hybrid_python_serial",
            "hybrid_native_1w", "hybrid_native_4w",
        ),
    )
    parser.add_argument("--dataset", required=True, choices=("landsat8", "rs"))
    parser.add_argument("--lambda-value", required=True, type=int)
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--warmup-images", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def byte_count(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, dict):
        return sum(byte_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(byte_count(item) for item in value)
    return 0


def sync(device):
    torch.cuda.synchronize(torch.device(device))


def timed(device, function):
    sync(device)
    start = time.perf_counter()
    output = function()
    sync(device)
    return output, time.perf_counter() - start


def load_model(args):
    if args.method in (
        "mamba_serial", "hybrid_1w", "hybrid_4w",
        "hybrid_batched_multistream", "hybrid_python_serial",
        "hybrid_native_1w", "hybrid_native_4w",
    ):
        project_root = Path("/data/huzhenghao/ml/h")
        sys.path.insert(0, str(project_root))
        channels = 7 if args.dataset == "landsat8" else 10
        if args.method == "mamba_serial":
            from model.mamba.mambamodel_1_917 import HNet

            wrapper = HNet(192, 320, channels)
            checkpoint_path = TEST_ROOT / "pth" / "our" / args.dataset / f"{args.lambda_value}.pth"
        else:
            sys.path.insert(0, str(OPT_ROOT / "model"))
            from mambamodel_hybrid_parallel import HNet

            python_serial = args.method == "hybrid_python_serial"
            one_worker = args.method in ("hybrid_1w", "hybrid_native_1w")
            wrapper = HNet(
                192,
                320,
                channels,
                entropy_workers=1 if one_worker or python_serial else 4,
                use_native_entropy=not python_serial,
            )
            mappings = {
                ("landsat8", 16384): OPT_ROOT / "experiments/train/20261019_0334_16384/bestmodel.pth",
                ("rs", 262144): OPT_ROOT / "experiments/RS/train/20261021_1946_262144/bestmodel.pth",
            }
            checkpoint_path = mappings[(args.dataset, args.lambda_value)]
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        remapped = {}
        pattern = re.compile(r"\._(matrix|bias|factor)(\d+)(?=\.|$)")
        plural = {"matrix": "matrices", "bias": "biases", "factor": "factors"}
        for key, value in state.items():
            match = pattern.search(key)
            if match:
                key = key[:match.start()] + f".{plural[match.group(1)]}.{match.group(2)}" + key[match.end():]
            remapped[key] = value
        wrapper.load_state_dict(remapped, strict=False)
        wrapper.eval().to(args.device)
        wrapper.net.update(force=True)
        return wrapper.net, channels

    if args.method == "rwk":
        import compressai

        expected_torch = (1, 12)
        actual_torch = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
        if actual_torch != expected_torch or compressai.__version__ != "1.2.6":
            raise RuntimeError(
                "RWK must be evaluated with its training environment: "
                "PyTorch 1.12 and CompressAI 1.2.6; "
                "use run_codec_benchmark.sh so its isolated environment is selected."
            )
        kernel_path = Path(
            "/data/huzhenghao/.cache/torch_extensions/py39_cu112/"
            "biwkv4/biwkv4.so"
        )
        if not kernel_path.is_file():
            raise FileNotFoundError(
                "The historical CUDA 11.2 RWK kernel is missing: "
                f"{kernel_path}"
            )
        spec = importlib.util.spec_from_file_location("biwkv4", kernel_path)
        kernel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kernel)
        sys.modules["biwkv4"] = kernel
        import model.rwk.models.lalic as rwk_model

        # Do not rebuild with the current system CUDA 13 toolchain. The cached
        # binary is the exact ABI/kernel path used by the historical training
        # environment and is required to reproduce the checkpoint numerics.
        rwk_model.load_biwkv4 = lambda: kernel

    from testmodel import getLoad, getNetByMethod

    channels = 7 if args.dataset == "landsat8" else 10
    model = getNetByMethod(args.method, 192, 320, channels)
    model = getLoad(model, args.method, args.dataset, args.lambda_value)
    if model is None:
        raise FileNotFoundError(f"Checkpoint missing: {args.method}/{args.dataset}/{args.lambda_value}")
    if args.method == "hpcm":
        # HPCM's compress() imports the same package by a second top-level name;
        # alias it to avoid registering the pybind classes twice.
        import model.hpcm.entropy_models as hpcm_entropy_models

        sys.modules["entropy_models"] = hpcm_entropy_models
        scale_table = torch.exp(torch.linspace(math.log(0.12), math.log(64.0), 60))
        model.update(scale_table)
    model.eval().to(args.device)
    if args.method not in ("hl", "hpcm") and hasattr(model, "update"):
        try:
            model.update(force=True)
        except TypeError:
            model.update()
    return model, channels


def codec_calls(method, model, image, temporary_path):
    if method.startswith("hybrid_"):
        encoded = lambda: model.compress(image, return_debug=True)
        decoded = lambda result: model.decompress(result["strings"], result["shape"], return_debug=True)
        size = lambda result: byte_count(result["strings"])
        exact = lambda result, output: torch.equal(result["debug_y_hat"], output["debug_y_hat"])
        return encoded, decoded, size, exact
    if method == "hl":
        from hl_real_codec_adapter import HLRealCodecAdapter

        codec = HLRealCodecAdapter(model)
        encoded = lambda: codec.compress(image)
        decoded = lambda result: codec.decompress(result["strings"], result["meta"])
        size = lambda result: byte_count(result["strings"])
        exact = lambda result, output: all(
            torch.equal(left, right)
            for left, right in zip(result["debug_y_hat"], output["debug_y_hat"])
        )
        return encoded, decoded, size, exact
    if method == "ftic":
        encoded = lambda: model.compress(image, str(temporary_path))
        decoded = lambda result: model.decompress(
            result["z_strings"], result["minmax"], result["z_shape"], str(temporary_path)
        )
        size = lambda result: int(result["y_size"]) + byte_count(result["z_strings"])
        exact = lambda result, output: torch.equal(
            result["debug_y_hat"], output["debug_y_hat"]
        )
        return encoded, decoded, size, exact
    encoded = lambda: model.compress(image)
    decoded = lambda result: model.decompress(result["strings"], result["shape"])
    size = lambda result: byte_count(result["strings"])
    # Most third-party codecs do not expose the encoder-side quantized latent,
    # so bit-exact latent verification is unavailable rather than implicitly
    # passing. Reconstruction and real bitstream decoding are still measured.
    exact = lambda result, output: None
    return encoded, decoded, size, exact


@torch.inference_mode()
def main():
    args = arguments()
    model, channels = load_model(args)
    dataset = TEST_ROOT / "testdata" / args.dataset / "baseall" / "val"
    paths = sorted(dataset.glob("*.npy"))[: args.warmup_images + args.num_images]
    normalization = 65535.0
    rows = []
    with tempfile.TemporaryDirectory(prefix=f"codec_{args.method}_") as temporary:
        stream_path = Path(temporary) / "stream.bin"
        for index, path in enumerate(paths):
            image = torch.from_numpy(np.load(path).astype(np.float32)).unsqueeze(0).to(args.device) / normalization
            encode, decode, size, exact = codec_calls(args.method, model, image, stream_path)
            result, encode_seconds = timed(args.device, encode)
            output, decode_seconds = timed(args.device, lambda: decode(result))
            if index < args.warmup_images:
                continue
            reconstruction = output["x_hat"].clamp(0.0, 1.0)
            mse = torch.mean((reconstruction - image) ** 2)
            similarity = ms_ssim(reconstruction, image, data_range=1.0)
            rows.append(
                {
                    "compress_s": encode_seconds,
                    "decompress_s": decode_seconds,
                    "bpp": size(result) * 8.0 / (image.shape[-2] * image.shape[-1]),
                    "psnr": float((-10 * torch.log10(mse)).item()),
                    "ms_ssim_db": float((-10 * torch.log10(1 - similarity)).item()),
                    "exact": exact(result, output),
                }
            )
            print(f"{args.method}/{args.dataset} {len(rows)}/{args.num_images}", flush=True)
    summary = {
        "method": args.method,
        "dataset": args.dataset,
        "lambda": args.lambda_value,
        "images": len(rows),
        "compress_mean_s": float(np.mean([row["compress_s"] for row in rows])),
        "compress_median_s": float(np.median([row["compress_s"] for row in rows])),
        "decompress_mean_s": float(np.mean([row["decompress_s"] for row in rows])),
        "decompress_median_s": float(np.median([row["decompress_s"] for row in rows])),
        "bpp": float(np.mean([row["bpp"] for row in rows])),
        "psnr": float(np.mean([row["psnr"] for row in rows])),
        "ms_ssim_db": float(np.mean([row["ms_ssim_db"] for row in rows])),
        "latent_exact": (
            all(row["exact"] for row in rows)
            if all(row["exact"] is not None for row in rows)
            else None
        ),
        "codec_note": (
            "independent checkerboard torchac adapter"
            if args.method == "hl"
            else "native vectorized rANS"
            if args.method == "ftic"
            else "native OpenMP multi-stream rANS"
            if args.method in ("hybrid_native_4w", "hybrid_batched_multistream", "hybrid_4w")
            else "native repository codec"
        ),
    }
    print("RESULT_JSON=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
