"""Plot the latest full-test RD curves for Landsat8 and RS.

The committed CSV contains the completed Landsat8/RS full-test comparison.
When the Landsat8 600-epoch pipeline has produced its final table, this script
automatically replaces only the Landsat8 Hybrid rows with that newer result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


RESULTS = Path(__file__).resolve().parent
OPT_ROOT = RESULTS.parent
BASE_CSV = RESULTS / "data" / "full_test_rd_all_methods.csv"
FINAL_600_CSV = OPT_ROOT / "final_results" / "landsat8_hybrid_600" / "landsat8_full_rd.csv"
FIGURES = RESULTS / "figures"

LABELS = {
    "elic": "ELIC",
    "rwk": "RWK",
    "hl": "HL",
    "mlic": "MLIC",
    "ftic": "FTIC",
    "hpcm": "HPCM",
    "our": "Our",
    "hybrid_parallel": "Hybrid-Parallel",
}

COLORS = {
    "elic": "#4477AA",
    "rwk": "#EE6677",
    "hl": "#228833",
    "mlic": "#AA3377",
    "ftic": "#66CCEE",
    "hpcm": "#888888",
    "our": "#332288",
    "hybrid_parallel": "#CC3311",
}

METHOD_ORDER = (
    "elic", "rwk", "hl", "mlic", "ftic", "hpcm", "our", "hybrid_parallel"
)
MARKERS = {
    "elic": "s", "rwk": "o", "hl": "^", "mlic": "v",
    "ftic": "P", "hpcm": "X", "our": "D", "hybrid_parallel": "*",
}
LINESTYLES = {"elic": "--", "hybrid_parallel": "-"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "axes.linewidth": 1.1,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def load_latest_results() -> tuple[pd.DataFrame, str]:
    data = pd.read_csv(BASE_CSV)
    source = "committed full-test CSV (Landsat8 400 epochs; RS completed run)"
    if FINAL_600_CSV.is_file():
        final = pd.read_csv(FINAL_600_CSV)
        required = {"Dataset", "Method", "bpp", "psnr", "ms_ssim_db"}
        if required.issubset(final.columns):
            replacement = final[
                final["Dataset"].str.lower().eq("landsat8")
                & final["Method"].str.lower().eq("hybrid_parallel")
            ]
            if len(replacement) == 7 and replacement["images"].min() == 1700:
                keep = ~(
                    data["Dataset"].str.lower().eq("landsat8")
                    & data["Method"].str.lower().eq("hybrid_parallel")
                )
                data = pd.concat([data[keep], replacement], ignore_index=True)
                source = "Landsat8 Hybrid 600-epoch final CSV + committed comparison CSV"
    return data, source


def hybrid_tuples(data: pd.DataFrame, dataset: str) -> list[tuple[float, float, float]]:
    rows = data[
        data["Dataset"].str.lower().eq(dataset)
        & data["Method"].str.lower().eq("hybrid_parallel")
    ].sort_values("bpp")
    return [
        (round(row.bpp, 6), round(row.psnr, 6), round(row.ms_ssim_db, 6))
        for row in rows.itertuples(index=False)
    ]


def bd_rate(reference: pd.DataFrame, tested: pd.DataFrame, metric: str) -> tuple[float, float, float]:
    """Return ELIC-relative BD-Rate; negative means tested saves bitrate."""

    def curve(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        selected = frame[[metric, "bpp"]].dropna()
        selected = selected[selected["bpp"] > 0].sort_values(metric)
        selected = selected.groupby(metric, as_index=False)["bpp"].mean()
        return selected[metric].to_numpy(), np.log(selected["bpp"].to_numpy())

    x_reference, y_reference = curve(reference)
    x_tested, y_tested = curve(tested)
    low = max(float(x_reference.min()), float(x_tested.min()))
    high = min(float(x_reference.max()), float(x_tested.max()))
    if high <= low or len(x_reference) < 2 or len(x_tested) < 2:
        return float("nan"), low, high
    reference_curve = PchipInterpolator(x_reference, y_reference)
    tested_curve = PchipInterpolator(x_tested, y_tested)
    reference_average = reference_curve.integrate(low, high) / (high - low)
    tested_average = tested_curve.integrate(low, high) / (high - low)
    return float((np.exp(tested_average - reference_average) - 1.0) * 100.0), low, high


def calculate_bd_rate_vs_elic(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in ("landsat8", "rs"):
        subset = data[data["Dataset"].str.lower().eq(dataset)]
        reference = subset[subset["Method"].str.lower().eq("elic")]
        for method, tested in subset.groupby("Method"):
            psnr_rate, psnr_low, psnr_high = bd_rate(reference, tested, "psnr")
            ms_rate, ms_low, ms_high = bd_rate(reference, tested, "ms_ssim_db")
            rows.append(
                {
                    "dataset": dataset,
                    "reference": "ELIC",
                    "method": LABELS.get(method.lower(), method),
                    "bd_rate_psnr_percent": psnr_rate,
                    "psnr_overlap_low_db": psnr_low,
                    "psnr_overlap_high_db": psnr_high,
                    "bd_rate_ms_ssim_percent": ms_rate,
                    "ms_ssim_overlap_low_db": ms_low,
                    "ms_ssim_overlap_high_db": ms_high,
                    "interpretation": "negative = bitrate saving relative to ELIC",
                }
            )
    return pd.DataFrame(rows)


def draw_panel(ax, subset: pd.DataFrame, metric: str, title: str) -> None:
    methods = {method.lower(): group for method, group in subset.groupby("Method")}
    for key in METHOD_ORDER:
        if key not in methods:
            continue
        group = methods[key].sort_values("bpp")
        is_hybrid = key == "hybrid_parallel"
        is_reference = key == "elic"
        ax.plot(
            group["bpp"],
            group[metric],
            label=LABELS.get(key, key),
            marker=MARKERS[key],
            linestyle=LINESTYLES.get(key, "-"),
            markersize=11 if is_hybrid else 6.2,
            markeredgecolor="white" if is_hybrid else COLORS[key],
            markeredgewidth=0.9 if is_hybrid else 0.6,
            linewidth=2.8 if is_hybrid else (2.1 if is_reference else 1.8),
            color=COLORS[key],
            alpha=1.0 if is_hybrid or is_reference else 0.93,
            zorder=6 if is_hybrid else (5 if is_reference else 2),
        )
    ax.set_xlabel("Bits per pixel (BPP)")
    ax.set_ylabel("PSNR (dB)" if metric == "psnr" else "MS-SSIM (dB)")
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    x_max = float(subset["bpp"].max())
    y_min, y_max = float(subset[metric].min()), float(subset[metric].max())
    y_pad = max((y_max - y_min) * 0.055, 0.15)
    ax.set_xlim(0, x_max * 1.04)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.grid(color="#D8D8D8", linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(direction="out", length=4.5, width=0.9, colors="#333333")


def plot_dataset(data: pd.DataFrame, dataset: str, title: str, show: bool = False) -> None:
    subset = data[data["Dataset"].str.lower().eq(dataset)].copy()
    for metric, ylabel, suffix in (
        ("psnr", "PSNR (dB)", "psnr_bpp"),
        ("ms_ssim_db", "MS-SSIM (dB)", "msssim_bpp"),
    ):
        fig, ax = plt.subplots(figsize=(9.2, 5.4))
        draw_panel(ax, subset, metric, f"{title}: {ylabel} vs. BPP")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.015, 0.5),
            frameon=False,
            borderaxespad=0,
            handlelength=2.4,
            labelspacing=0.9,
        )
        fig.subplots_adjust(left=0.105, right=0.79, bottom=0.14, top=0.90)
        output = FIGURES / f"{dataset}_{suffix}_latest.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    data, source = load_latest_results()
    print("Data source:", source)
    for dataset in ("landsat8", "rs"):
        print(f'"{dataset}_hybrid_parallel": [')
        for row in hybrid_tuples(data, dataset):
            print(f"    {row},")
        print("]")
    bd_rates = calculate_bd_rate_vs_elic(data)
    bd_rates.to_csv(RESULTS / "data" / "bd_rate_vs_elic.csv", index=False)
    print("\nBD-Rate relative to ELIC (negative is better):")
    print(
        bd_rates[["dataset", "method", "bd_rate_psnr_percent", "bd_rate_ms_ssim_percent"]]
        .round(3)
        .to_string(index=False)
    )
    plot_dataset(data, "landsat8", "Landsat8", args.show)
    plot_dataset(data, "rs", "RS", args.show)


if __name__ == "__main__":
    main()
