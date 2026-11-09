"""Run all Landsat8 or RS hybrid checkpoints on the /ml/test test split."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

RUNS = {
    "landsat8": {
        "dataset": Path("/data/huzhenghao/ml/test/testdata/landsat8/baseall/val"),
        "channels": 7,
        "root": ROOT / "experiments" / "train",
        "mapping": {
            1024: "20261018_2331_1024",
            2048: "20261019_0741_2048",
            4096: "20261019_1638_4096",
            8192: "20261019_2052_8192",
            16384: "20261019_0334_16384",
            32768: "20261019_1145_32768",
            65536: "20261020_0055_65536",
        },
    },
    "rs": {
        "dataset": Path("/data/huzhenghao/ml/test/testdata/rs/baseall/val"),
        "channels": 10,
        "root": ROOT / "experiments" / "RS" / "train",
        "mapping": {
            32768: "20261020_1141_32768",
            65536: "20261020_2222_65536",
            131071: "20261021_0904_131071",
            262144: "20261021_1946_262144",
            393216: "20261022_0629_393216",
            524288: "20261022_1711_524288",
        },
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=tuple(RUNS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    spec = RUNS[args.dataset]
    output_root = ROOT / "experiments" / "full_test_comparison" / args.dataset / "hybrid"
    for lambda_value, run_name in spec["mapping"].items():
        output = output_root / str(lambda_value)
        checkpoint = spec["root"] / run_name / "bestmodel.pth"
        command = [
            args.python,
            str(ROOT / "evaluate_hybrid_full_test.py"),
            "--checkpoint", str(checkpoint),
            "--dataset", str(spec["dataset"]),
            "--output-dir", str(output),
            "--dataset-name", args.dataset,
            "--lambda-value", str(lambda_value),
            "--channels", str(spec["channels"]),
            "--normalization", "65535",
            "--device", args.device,
        ]
        print("RUN", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
