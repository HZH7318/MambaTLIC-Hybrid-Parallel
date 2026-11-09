"""Sequentially train hybrid-parallel models from a directory of INI files."""

import argparse
import subprocess
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = EXPERIMENT_ROOT / "config" / "auto_to_train"
TRAIN_SCRIPT = EXPERIMENT_ROOT / "run_mamba_hybrid_parallel.py"
DEFAULT_PYTHON = Path("/data/huzhenghao/anaconda3/envs/mambah/bin/python")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Directory containing the new architecture's .ini files.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_PYTHON,
        help="Python interpreter used to launch training jobs.",
    )
    parser.add_argument(
        "--lambdas",
        type=int,
        nargs="+",
        help="Only run these lambda configs, in ascending numeric order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config_dir.is_dir():
        raise FileNotFoundError(f"Config directory does not exist: {args.config_dir}")
    if not args.python.is_file():
        raise FileNotFoundError(f"Python interpreter does not exist: {args.python}")

    config_files = sorted(
        args.config_dir.glob("*.ini"), key=lambda path: int(path.stem)
    )
    if not config_files:
        raise FileNotFoundError(f"No .ini files found in: {args.config_dir}")
    if args.lambdas:
        requested = set(args.lambdas)
        available = {int(path.stem): path for path in config_files}
        missing = sorted(requested - available.keys())
        if missing:
            raise FileNotFoundError(f"Missing lambda config files: {missing}")
        config_files = [available[value] for value in sorted(requested)]

    for config_path in config_files:
        print("=" * 50)
        print(f"开始训练混合并行模型: {config_path}")
        print("=" * 50)
        subprocess.run(
            [str(args.python), str(TRAIN_SCRIPT), "--config", str(config_path)],
            check=True,
        )


if __name__ == "__main__":
    main()
