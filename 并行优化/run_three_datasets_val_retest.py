"""Run the unified validation benchmark in each method's proven environment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKER = ROOT / "evaluate_three_datasets_val.py"
SUMMARY = ROOT / "results" / "data" / "three_datasets_val_rd.csv"
LOG = ROOT / "results" / "val_retest" / "run.log"

METHOD_ENV = {
    "elic": "elic",
    "rwk": "nl3c",
    "hl": "nl3c",
    "ftic": "nl3c",
    "hpcm": "nl3c",
    "mlic": "mlic",
    "our": "mambah",
    "hybrid_parallel": "mambah",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--datasets", nargs="+", default=["landsat8", "rs", "fmow8"]
    )
    parser.add_argument("--methods", nargs="+", default=list(METHOD_ENV))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    with LOG.open("a", encoding="utf-8") as log:
        for method in args.methods:
            environment = METHOD_ENV[method]
            python = Path("/data/huzhenghao/anaconda3/envs") / environment / "bin/python"
            command = [
                str(python), "-u", str(WORKER),
                "--datasets", *args.datasets,
                "--methods", method,
                "--device", args.device,
                "--summary-csv", str(SUMMARY),
            ]
            if args.force:
                command.append("--force")
            print(f"RUN {method} in {environment}", flush=True)
            log.write(f"\nRUN {' '.join(command)}\n")
            log.flush()
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                failures.append((method, completed.returncode))
                print(f"FAILED {method}: exit={completed.returncode}", file=sys.stderr)
            else:
                print(f"DONE {method}", flush=True)
    if failures:
        raise SystemExit(f"Method failures: {failures}; see {LOG}")
    print(f"Summary: {SUMMARY}")


if __name__ == "__main__":
    main()
