"""Resume the seven corrected FMoW-4096 Hybrid runs to 300 epochs on one GPU.

The 150-epoch runs remain untouched.  Each continuation gets its own run
directory and starts from the highest saved epoch available for that lambda.
"""

from __future__ import annotations

import argparse
import configparser
import os
import subprocess
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = (
    ROOT
    / "experiments/FMoW_4096_representative_1000_300_150/train"
)
OUTPUT_ROOT = (
    ROOT
    / "experiments/FMoW_4096_representative_1000_300_300/train"
)
CONFIG_TEMPLATE = (
    ROOT
    / "config/fmow_4096_representative_1000_300_auto_to_train/1024.ini"
)
TRAIN_ENTRY = ROOT / "run_mamba_hybrid_parallel_fmow_4096.py"
PYTHON = Path("/data/huzhenghao/anaconda3/envs/mambah/bin/python")
LAMBDAS = (1024, 2048, 4096, 8192, 16384, 32768, 65536)


def checkpoint_epoch(path: Path) -> int:
    if path.stem.isdigit():
        return int(path.stem)
    checkpoint = torch.load(path, map_location="cpu")
    return int(checkpoint.get("epoch", -1))


def source_run(lmbda: int) -> Path:
    candidates = sorted(SOURCE_ROOT.glob(f"*_{lmbda}"))
    # Exclude the one aborted lambda=1024 run at epoch 0.
    usable = [
        run
        for run in candidates
        if (run / "bestmodel.pth").is_file()
        and checkpoint_epoch(run / "bestmodel.pth") > 0
    ]
    if not usable:
        raise FileNotFoundError(f"No completed 150-epoch run for lambda={lmbda}")
    return max(usable, key=lambda path: path.stat().st_mtime)


def prepare_resume_dir(lmbda: int) -> tuple[Path, int]:
    destination = OUTPUT_ROOT / f"lambda_{lmbda}"
    destination.mkdir(parents=True, exist_ok=True)

    existing = list(destination.glob("*.pth"))
    if existing:
        latest = max(existing, key=checkpoint_epoch)
        return destination, checkpoint_epoch(latest)

    run = source_run(lmbda)
    candidates = list(run.glob("*.pth"))
    latest = max(candidates, key=checkpoint_epoch)
    epoch = checkpoint_epoch(latest)
    link = destination / f"resume_from_epoch_{epoch}.pth"
    link.symlink_to(latest.resolve())
    return destination, epoch


def make_config(lmbda: int, device: str, resume_dir: Path) -> Path:
    config = configparser.ConfigParser()
    config.read(CONFIG_TEMPLATE)
    train = config["trainconfig"]
    test = config["testconfig"]
    train["lmbda"] = str(lmbda)
    test["lmbda"] = str(lmbda)
    train["device"] = device
    test["device"] = device
    train["train_max_epoch"] = "300"
    train["continue_train"] = "True"
    train["pretrained_path"] = str(resume_dir)
    train["train_log"] = str(OUTPUT_ROOT)
    train["info"] = "hybrid_parallel_tail_fmow_4096_rep1000_resume_to_300"
    test["info"] = train["info"]
    test["test_model_path"] = str(resume_dir)

    config_dir = ROOT / "config/fmow_4096_representative_1000_300_resume_300"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{lmbda}.ini"
    with path.open("w", encoding="utf-8") as handle:
        config.write(handle)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lambdas", nargs="+", type=int, default=LAMBDAS)
    args = parser.parse_args()

    selected = tuple(sorted(set(args.lambdas)))
    unsupported = sorted(set(selected) - set(LAMBDAS))
    if unsupported:
        raise ValueError(f"Unsupported lambdas: {unsupported}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for lmbda in selected:
        resume_dir, epoch = prepare_resume_dir(lmbda)
        config = make_config(lmbda, args.device, resume_dir)
        print(
            f"Resuming FMoW lambda={lmbda} from saved epoch={epoch} "
            f"to epoch=299 on {args.device}",
            flush=True,
        )
        subprocess.run(
            [str(PYTHON), "-u", str(TRAIN_ENTRY), "--config", str(config)],
            cwd=ROOT,
            env=os.environ.copy(),
            check=True,
        )


if __name__ == "__main__":
    main()
