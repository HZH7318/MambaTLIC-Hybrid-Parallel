"""Measure synchronized stages of the real Hybrid compress/decompress path.

The profiler is intentionally opt-in.  Normal codec calls do not synchronize
between stages and therefore keep their original execution behavior.  This
script targets the P2 path: searchsorted scale indexes plus one deferred
14-stream native rANS submission on the encoder.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from benchmark_codec_time_method import byte_count, load_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        default="hybrid_searchsorted_14stream_4w",
        choices=(
            "hybrid_searchsorted_14stream_1w",
            "hybrid_searchsorted_14stream_4w",
            "hybrid_searchsorted_fusedpack_4w",
        ),
    )
    parser.add_argument("--dataset", default="landsat8", choices=("landsat8", "rs"))
    parser.add_argument("--lambda-value", type=int, default=16384)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-images", type=int, default=5)
    parser.add_argument("--num-images", type=int, default=30)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def synchronized_call(device, function):
    torch.cuda.synchronize(torch.device(device))
    started = time.perf_counter()
    value = function()
    torch.cuda.synchronize(torch.device(device))
    return value, time.perf_counter() - started


def means(rows):
    keys = sorted({key for row in rows for key in row})
    return {key: statistics.fmean(row.get(key, 0.0) for row in rows) for key in keys}


@torch.inference_mode()
def main():
    args = parse_args()
    method = args.method
    model_args = SimpleNamespace(
        method=method,
        dataset=args.dataset,
        lambda_value=args.lambda_value,
        device=args.device,
    )
    model, _ = load_model(model_args)
    data_root = (
        Path("/data/huzhenghao/ml/test/testdata")
        / args.dataset
        / "baseall"
        / "val"
    )
    paths = sorted(data_root.glob("*.npy"))
    required = args.warmup_images + args.num_images
    if len(paths) < required:
        raise RuntimeError(f"Need {required} images, found {len(paths)} in {data_root}")

    encode_rows = []
    decode_rows = []
    complete_encode = []
    complete_decode = []
    stream_bytes = []
    for index, path in enumerate(paths[:required]):
        image = (
            torch.from_numpy(np.load(path).astype(np.float32))
            .unsqueeze(0)
            .to(args.device)
            / 65535.0
        )
        measured = index >= args.warmup_images
        encoded, encode_s = synchronized_call(
            args.device,
            lambda: model.compress(image, profile_stages=measured),
        )
        decoded, decode_s = synchronized_call(
            args.device,
            lambda: model.decompress(
                encoded["strings"], encoded["shape"], profile_stages=measured
            ),
        )
        if not measured:
            continue
        encode_rows.append(encoded["stage_profile"])
        decode_rows.append(decoded["stage_profile"])
        complete_encode.append(encode_s)
        complete_decode.append(decode_s)
        stream_bytes.append(byte_count(encoded["strings"]))
        print(f"{index - args.warmup_images + 1}/{args.num_images}", flush=True)

    encode_mean = means(encode_rows)
    decode_mean = means(decode_rows)
    native_detail_keys = {
        "y_symbol_quantization_packing",
        "y_device_to_host",
        "y_host_views_and_cdf",
        "y_native_rans_encode",
    }
    encode_stage_sum = sum(
        value
        for key, value in encode_mean.items()
        if key != "profiled_total" and key not in native_detail_keys
    )
    encode_native_detail_sum = sum(
        encode_mean.get(key, 0.0) for key in native_detail_keys
    )
    decode_stage_sum = sum(
        value for key, value in decode_mean.items() if key != "profiled_total"
    )
    result = {
        "method": method,
        "dataset": args.dataset,
        "lambda": args.lambda_value,
        "warmup_images": args.warmup_images,
        "images": args.num_images,
        "compress_mean_s": statistics.fmean(complete_encode),
        "compress_median_s": statistics.median(complete_encode),
        "decompress_mean_s": statistics.fmean(complete_decode),
        "decompress_median_s": statistics.median(complete_decode),
        "stream_bytes_mean": statistics.fmean(stream_bytes),
        "encode_stage_mean_s": encode_mean,
        "decode_stage_mean_s": decode_mean,
        "encode_stage_sum_s": encode_stage_sum,
        "encode_native_detail_sum_s": encode_native_detail_sum,
        "decode_stage_sum_s": decode_stage_sum,
        "encode_unaccounted_s": statistics.fmean(complete_encode) - encode_stage_sum,
        "decode_unaccounted_s": statistics.fmean(complete_decode) - decode_stage_sum,
    }
    if args.output is None:
        args.output = (
            ROOT
            / "experiments"
            / "codec_stage_profile"
            / f"{args.dataset}_{args.lambda_value}_p2_4w.json"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("RESULT_JSON=" + json.dumps(result), flush=True)
    print(f"OUTPUT={args.output}", flush=True)


if __name__ == "__main__":
    main()
