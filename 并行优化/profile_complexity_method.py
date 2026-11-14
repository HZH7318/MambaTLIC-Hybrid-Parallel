"""Profile one codec with a reproducible model-complexity protocol.

Params are available on CPU.  Peak memory and FLOPs require a real forward
pass and should be measured on an otherwise idle GPU.  The script deliberately
does not estimate missing values from checkpoint size or from another paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace

import torch

from benchmark_codec_time_method import load_model


METHOD_ALIASES = {
    "hybrid_parallel": "hybrid_native_4w",
    "our_original": "mamba_serial",
    "our_native_1t": "hybrid_native_1w",
}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=(
            "elic", "rwk", "mlic", "hpcm", "ftic", "hl",
            "mamba_serial", "hybrid_parallel", "our_original", "our_native_1t",
        ),
    )
    parser.add_argument("--dataset", required=True, choices=("landsat8", "rs"))
    parser.add_argument("--lambda-value", required=True, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--params-only", action="store_true")
    return parser.parse_args()


def parameter_statistics(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


@torch.inference_mode()
def profile_forward(model, image, device):
    if device.type != "cuda":
        raise ValueError("Mem(G) must be measured on CUDA; use --params-only on CPU")

    # Warm up kernels and one-off workspaces before resetting peak statistics.
    model(image)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_bytes = torch.cuda.memory_allocated(device)
    model(image)
    torch.cuda.synchronize(device)
    peak_bytes = torch.cuda.max_memory_allocated(device)

    try:
        try:
            from thop import profile
        except ImportError:
            # RWK is intentionally kept in its historical PyTorch 1.12
            # environment. THOP is pure Python, so reuse the base installation
            # without changing that frozen environment.
            sys.path.append("/data/huzhenghao/anaconda3/lib/python3.9/site-packages")
            from thop import profile

        macs, _ = profile(model, inputs=(image,), verbose=False)
        # THOP reports MACs.  We expose both the native MAC count and the
        # common arithmetic convention 1 MAC = 2 FLOPs.
        macs_g = float(macs) / 1e9
        flops_g = 2.0 * macs_g
        flops_status = "thop_supported_ops_only_1mac_2flops"
    except Exception as error:  # Keep valid memory/parameter measurements.
        macs_g = None
        flops_g = None
        flops_status = f"unavailable:{type(error).__name__}:{error}"

    return {
        "model_baseline_mem_gib": baseline_bytes / (1024 ** 3),
        "forward_peak_mem_gib": peak_bytes / (1024 ** 3),
        "forward_activation_delta_gib": (peak_bytes - baseline_bytes) / (1024 ** 3),
        "macs_g": macs_g,
        "flops_g": flops_g,
        "flops_status": flops_status,
    }


def main():
    args = arguments()
    method_for_loader = METHOD_ALIASES.get(args.method, args.method)
    loader_args = SimpleNamespace(
        method=method_for_loader,
        dataset=args.dataset,
        lambda_value=args.lambda_value,
        device=args.device,
    )
    model, channels = load_model(loader_args)
    total, trainable = parameter_statistics(model)
    result = {
        "dataset": args.dataset,
        "method": args.method,
        "lambda": args.lambda_value,
        "input_shape": [1, channels, args.height, args.width],
        "params_m": total / 1e6,
        "trainable_params_m": trainable / 1e6,
        "forward_peak_mem_gib": None,
        "flops_g": None,
        "protocol": "forward_inference_batch1_256x256",
    }
    if not args.params_only:
        device = torch.device(args.device)
        image = torch.rand(
            1, channels, args.height, args.width, device=device, dtype=torch.float32
        )
        result.update(profile_forward(model, image, device))
    print("COMPLEXITY_JSON=" + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
