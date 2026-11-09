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
    "elic": "#111111",
    "rwk": "#E69F00",
    "hl": "#009E73",
    "mlic": "#CC79A7",
    "ftic": "#56B4E9",
    "hpcm": "#7F7F7F",
    "our": "#7B2CBF",
    "hybrid_parallel": "#D62728",
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
        "font.family": "DejaVu Serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.linewidth": 1.0,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "savefig.facecolor": "white",
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
            markersize=10 if is_hybrid else 5.5,
            markeredgecolor="white" if is_hybrid else COLORS[key],
            markeredgewidth=0.7,
            linewidth=2.7 if is_hybrid else (2.2 if is_reference else 1.7),
            color=COLORS[key],
            zorder=6 if is_hybrid else (5 if is_reference else 2),
        )
    ax.set_xlabel("Bits per pixel (BPP)")
    ax.set_ylabel("PSNR (dB)" if metric == "psnr" else "MS-SSIM (dB)")
    ax.set_title(title, fontweight="semibold")
    ax.set_xlim(left=0)
    ax.grid(which="major", color="#C8C8C8", linewidth=0.65, alpha=0.55)
    ax.minorticks_on()
    ax.grid(which="minor", color="#E8E8E8", linewidth=0.35, alpha=0.35)
    ax.tick_params(direction="in", top=True, right=True)


def plot_dataset(data: pd.DataFrame, dataset: str, title: str, show: bool = False) -> None:
    subset = data[data["Dataset"].str.lower().eq(dataset)].copy()
    for metric, ylabel, suffix in (
        ("psnr", "PSNR (dB)", "psnr_bpp"),
        ("ms_ssim_db", "MS-SSIM (dB)", "msssim_bpp"),
    ):
        fig, ax = plt.subplots(figsize=(8.2, 5.4))
        draw_panel(ax, subset, metric, f"{title}: {ylabel} vs. BPP")
        ax.legend(ncol=2, frameon=True, framealpha=0.92, edgecolor="#BBBBBB")
        fig.tight_layout()
        output = FIGURES / f"{dataset}_{suffix}_latest.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


def plot_publication_grid(data: pd.DataFrame, show: bool = False) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))
    panels = (
        ("landsat8", "psnr", "(a) Landsat8: PSNR"),
        ("landsat8", "ms_ssim_db", "(b) Landsat8: MS-SSIM"),
        ("rs", "psnr", "(c) RS: PSNR"),
        ("rs", "ms_ssim_db", "(d) RS: MS-SSIM"),
    )
    for ax, (dataset, metric, title) in zip(axes.flat, panels):
        subset = data[data["Dataset"].str.lower().eq(dataset)]
        draw_panel(ax, subset, metric, title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=8,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=1.3,
        handletextpad=0.45,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=2.2, w_pad=2.0)
    fig.savefig(FIGURES / "landsat8_rs_rd_publication.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "landsat8_rs_rd_publication.pdf", bbox_inches="tight")
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
    plot_publication_grid(data, args.show)


if __name__ == "__main__":
    main()
