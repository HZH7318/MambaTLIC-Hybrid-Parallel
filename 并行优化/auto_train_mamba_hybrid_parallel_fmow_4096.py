"""Run the corrected FMoW-4096 Hybrid configurations sequentially."""

import argparse
import configparser
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = ROOT / "config" / "fmow_4096_representative_1000_300_auto_to_train"
TRAIN_SCRIPT = ROOT / "run_mamba_hybrid_parallel_fmow_4096.py"
DEFAULT_PYTHON = Path("/data/huzhenghao/anaconda3/envs/mambah/bin/python")
SUPPORTED_LAMBDAS = (1024, 2048, 4096, 8192, 16384, 32768, 65536)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--lambdas", type=int, nargs="+")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument(
        "--device",
        type=str,
        help="Override train/test device from the template, e.g. cuda:1.",
    )
    args = parser.parse_args()

    template = args.config_dir / "1024.ini"
    if not template.is_file():
        raise FileNotFoundError(f"Missing FMoW template config: {template}")
    selected = sorted(set(args.lambdas or SUPPORTED_LAMBDAS))
    missing = sorted(set(selected) - set(SUPPORTED_LAMBDAS))
    if missing:
        raise ValueError(f"Unsupported FMoW lambdas: {missing}")

    with tempfile.TemporaryDirectory(prefix="fmow_4096_configs_") as temp_dir:
        for value in selected:
            parser_config = configparser.ConfigParser()
            parser_config.read(template)
            parser_config["trainconfig"]["lmbda"] = str(value)
            parser_config["testconfig"]["lmbda"] = str(value)
            if args.device is not None:
                parser_config["trainconfig"]["device"] = args.device
                parser_config["testconfig"]["device"] = args.device
            if args.max_epochs is not None:
                parser_config["trainconfig"]["train_max_epoch"] = str(args.max_epochs)
            config = Path(temp_dir) / f"{value}.ini"
            with config.open("w", encoding="utf-8") as handle:
                parser_config.write(handle)
            print(
                f"Starting corrected FMoW-4096 Hybrid model: lambda={value}, "
                f"config={config}",
                flush=True,
            )
            subprocess.run(
                [str(args.python), "-u", str(TRAIN_SCRIPT), "--config", str(config)],
                check=True,
            )


if __name__ == "__main__":
    main()
