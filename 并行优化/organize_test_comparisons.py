"""Organize /ml/test results for Landsat8 and RS without changing old files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


TEST_ROOT = Path("/data/huzhenghao/ml/test")
ROOT = Path(__file__).resolve().parent
HYBRID_ROOT = ROOT / "experiments" / "full_test_comparison"
OUT = ROOT / "experiments" / "comparison_all_methods"
DATASETS = ("landsat8", "rs")


def aggregate_existing() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(TEST_ROOT / "summary_all_results.csv")
    raw = raw[raw["Dataset"].isin(DATASETS)].copy()
    grouped = (
        raw.groupby(["Dataset", "Method", "Lambda"], as_index=False)
        .agg(
            images=("FileName", "size"),
            unique_images=("FileName", "nunique"),
            psnr=("PSNR", "mean"),
            ms_ssim_db=("MS-SSIM", "mean"),
            bpp=("BPP", "mean"),
            psnr_std=("PSNR", "std"),
            ms_ssim_db_std=("MS-SSIM", "std"),
            bpp_std=("BPP", "std"),
        )
        .sort_values(["Dataset", "Method", "bpp"])
    )
    return raw, grouped


def collect_hybrid() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        root = HYBRID_ROOT / dataset / "hybrid"
        if not root.exists():
            continue
        for path in sorted(root.glob("*/summary.json")):
            summary = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "Dataset": dataset,
                    "Method": "hybrid_parallel",
                    "Lambda": summary["lambda"],
                    "images": summary["num_images"],
                    "unique_images": summary["num_images"],
                    "psnr": summary["metrics"]["psnr"]["mean"],
                    "ms_ssim_db": summary["metrics"]["ms_ssim_db"]["mean"],
                    "bpp": summary["metrics"]["bpp"]["mean"],
                    "psnr_std": summary["metrics"]["psnr"]["std"],
                    "ms_ssim_db_std": summary["metrics"]["ms_ssim_db"]["std"],
                    "bpp_std": summary["metrics"]["bpp"]["std"],
                }
            )
    return pd.DataFrame(rows)


def monotonic_audit(grouped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method), group in grouped.groupby(["Dataset", "Method"]):
        by_lambda = group.sort_values("Lambda")
        previous = None
        for row in by_lambda.itertuples(index=False):
            if previous is not None:
                violations = []
                if row.bpp < previous.bpp:
                    violations.append("BPP decreases")
                if row.psnr < previous.psnr:
                    violations.append("PSNR decreases")
                if row.ms_ssim_db < previous.ms_ssim_db:
                    violations.append("MS-SSIM decreases")
                if violations:
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "previous_lambda": int(previous.Lambda),
                            "lambda": int(row.Lambda),
                            "issues": "; ".join(violations),
                            "previous_bpp": previous.bpp,
                            "bpp": row.bpp,
                            "previous_psnr": previous.psnr,
                            "psnr": row.psnr,
                        }
                    )
            previous = row
    return pd.DataFrame(rows)


def protocol_audit(grouped: pd.DataFrame) -> pd.DataFrame:
    capabilities = {
        "elic": ("yes", "custom compress/decompress"),
        "rwk": ("yes", "inherits CompressAI Elic2022Official codec API"),
        "mlic": ("yes", "custom compress/decompress"),
        "ftic": ("adapter", "real file-based coder; needs isolated temporary output per image"),
        "hpcm": ("yes", "custom compress/decompress and rANS extension"),
        "hl": ("no", "forward only; BPP is likelihood estimate"),
        "our": ("yes", "original Mamba compress/decompress"),
        "hybrid_parallel": ("yes", "independent real tail streams"),
    }
    rows = []
    methods = sorted(set(grouped["Method"]).union({"hybrid_parallel"}))
    for method in methods:
        support, note = capabilities.get(method, ("unknown", "manual audit needed"))
        rows.append(
            {
                "method": method,
                "full_forward_rd_available": method != "hybrid_parallel" or not collect_hybrid().empty,
                "real_codec_supported": support,
                "current_out_time_valid_as_real_codec": False,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def bd_rate(reference: pd.DataFrame, tested: pd.DataFrame, metric: str) -> tuple[float, float, float]:
    """Bjontegaard delta-rate; negative means tested uses fewer bits."""
    def curve(frame: pd.DataFrame):
        selected = frame[[metric, "bpp"]].dropna().sort_values(metric)
        selected = selected.groupby(metric, as_index=False).bpp.mean()
        return selected[metric].to_numpy(), np.log(selected.bpp.to_numpy())

    x_ref, y_ref = curve(reference)
    x_test, y_test = curve(tested)
    low = max(float(x_ref.min()), float(x_test.min()))
    high = min(float(x_ref.max()), float(x_test.max()))
    if high <= low or len(x_ref) < 2 or len(x_test) < 2:
        return float("nan"), low, high
    ref_interp = PchipInterpolator(x_ref, y_ref)
    test_interp = PchipInterpolator(x_test, y_test)
    ref_average = ref_interp.integrate(low, high) / (high - low)
    test_average = test_interp.integrate(low, high) / (high - low)
    return float((np.exp(test_average - ref_average) - 1.0) * 100.0), low, high


def calculate_hybrid_bd_rates(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        subset = combined[combined["Dataset"] == dataset]
        hybrid = subset[subset["Method"] == "hybrid_parallel"]
        if len(hybrid) < 4:
            continue
        for method, reference in subset[subset["Method"] != "hybrid_parallel"].groupby("Method"):
            psnr_rate, psnr_low, psnr_high = bd_rate(reference, hybrid, "psnr")
            ms_rate, ms_low, ms_high = bd_rate(reference, hybrid, "ms_ssim_db")
            rows.append(
                {
                    "dataset": dataset,
                    "reference_method": method,
                    "tested_method": "hybrid_parallel",
                    "bd_rate_psnr_percent": psnr_rate,
                    "psnr_overlap_low_db": psnr_low,
                    "psnr_overlap_high_db": psnr_high,
                    "bd_rate_ms_ssim_percent": ms_rate,
                    "ms_ssim_overlap_low_db": ms_low,
                    "ms_ssim_overlap_high_db": ms_high,
                    "interpretation": "negative = hybrid bitrate saving",
                }
            )
    return pd.DataFrame(rows)


def plot_rd(data: pd.DataFrame) -> None:
    colors = {
        "elic": "#1f77b4", "rwk": "#ff7f0e", "hl": "#2ca02c",
        "mlic": "#9467bd", "ftic": "#8c564b", "hpcm": "#23201f",
        "our": "#e377c2", "hybrid_parallel": "#0eff6a",
    }
    for dataset in DATASETS:
        subset = data[data["Dataset"] == dataset]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for method, group in subset.groupby("Method"):
            group = group.sort_values("bpp")
            style = "*-" if method == "hybrid_parallel" else "o-"
            axes[0].plot(group.bpp, group.psnr, style, label=method, color=colors.get(method))
            axes[1].plot(group.bpp, group.ms_ssim_db, style, label=method, color=colors.get(method))
        axes[0].set(xlabel="BPP", ylabel="PSNR (dB)", title=f"{dataset}: full-test RD")
        axes[1].set(xlabel="BPP", ylabel="MS-SSIM (dB)", title=f"{dataset}: full-test RD")
        for axis in axes:
            axis.grid(alpha=0.3)
            axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"{dataset}_full_test_rd.png", dpi=180)
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw, existing = aggregate_existing()
    hybrid = collect_hybrid()
    combined = pd.concat([existing, hybrid], ignore_index=True) if not hybrid.empty else existing
    combined = combined.sort_values(["Dataset", "Method", "bpp"]).reset_index(drop=True)
    coverage = (
        existing.groupby(["Dataset", "Method"], as_index=False)
        .agg(lambda_points=("Lambda", "size"), min_images=("images", "min"), max_images=("images", "max"))
    )
    audit = monotonic_audit(existing)
    protocols = protocol_audit(combined)
    bd_rates = calculate_hybrid_bd_rates(combined)
    raw.to_csv(OUT / "full_test_per_image_existing_methods.csv", index=False)
    existing.to_csv(OUT / "full_test_rd_existing_methods.csv", index=False)
    combined.to_csv(OUT / "full_test_rd_with_hybrid_completed_so_far.csv", index=False)
    combined.to_csv(OUT / "full_test_rd_all_methods.csv", index=False)
    coverage.to_csv(OUT / "coverage.csv", index=False)
    audit.to_csv(OUT / "monotonicity_audit.csv", index=False)
    protocols.to_csv(OUT / "method_protocol_audit.csv", index=False)
    bd_rates.to_csv(OUT / "bd_rate_hybrid_vs_methods.csv", index=False)
    plot_rd(combined)
    print("Coverage:\n", coverage.to_string(index=False))
    print("\nMonotonicity warnings:\n", audit.to_string(index=False) if len(audit) else "none")
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
