"""Plot the latest full-test RD curves for Landsat8 and RS.

The committed CSV contains the completed Landsat8/RS full-test comparison.
When the Landsat8 600-epoch pipeline has produced its final table, this script
automatically replaces only the Landsat8 Hybrid rows with that newer result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
    "elic": "#1f77b4",
    "rwk": "#ff7f0e",
    "hl": "#2ca02c",
    "mlic": "#9467bd",
    "ftic": "#8c564b",
    "hpcm": "#23201f",
    "our": "#e377c2",
    "hybrid_parallel": "#00c853",
}


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


def plot_dataset(data: pd.DataFrame, dataset: str, title: str, show: bool = False) -> None:
    subset = data[data["Dataset"].str.lower().eq(dataset)].copy()
    for metric, ylabel, suffix in (
        ("psnr", "PSNR (dB)", "psnr_bpp"),
        ("ms_ssim_db", "MS-SSIM (dB)", "msssim_bpp"),
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        for method, group in subset.groupby("Method"):
            key = method.lower()
            group = group.sort_values("bpp")
            is_hybrid = key == "hybrid_parallel"
            ax.plot(
                group["bpp"],
                group[metric],
                label=LABELS.get(key, method),
                marker="*" if is_hybrid else "o",
                markersize=11 if is_hybrid else 6,
                linewidth=2.6 if is_hybrid else 1.8,
                color=COLORS.get(key),
                zorder=5 if is_hybrid else 2,
            )
        ax.set_xlabel("BPP")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} - {ylabel} vs BPP")
        ax.grid(alpha=0.3)
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        output = FIGURES / f"{dataset}_{suffix}_latest.png"
        fig.savefig(output, dpi=220, bbox_inches="tight")
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
    plot_dataset(data, "landsat8", "Landsat8", args.show)
    plot_dataset(data, "rs", "RS", args.show)


if __name__ == "__main__":
    main()
