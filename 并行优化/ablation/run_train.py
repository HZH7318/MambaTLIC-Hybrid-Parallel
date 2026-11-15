"""Independent training entry point for the S0/S1 structure ablation."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OPT_ROOT = ROOT / "并行优化"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OPT_ROOT))
sys.path.insert(0, str(OPT_ROOT / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Parse only the variant here.  run_mamba_1 owns --config and parses it during
# import, so remove the ablation-only option before importing that module.
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--variant", required=True, choices=("s0", "s1"))
known, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]

from model_serial_grouped import RateDistortion, S0Net, S1Net  # noqa: E402
import run_mamba_1 as training  # noqa: E402


training.model = S0Net if known.variant == "s0" else S1Net
training.Loss = RateDistortion

# The legacy trainer creates a timestamped directory below train_log.  Resolve
# the newest completed/intermediate run automatically so a 400-epoch ablation
# can survive tmux interruption without editing the committed INI file.
train_root = Path(training.train_config["train_log"])
checkpoint_dirs = []
if train_root.is_dir():
    for directory in train_root.iterdir():
        if directory.is_dir() and glob.glob(str(directory / "*.pth")):
            checkpoint_dirs.append(directory)
if checkpoint_dirs:
    latest = max(
        checkpoint_dirs,
        key=lambda directory: max(path.stat().st_mtime for path in directory.glob("*.pth")),
    )
    training.train_config["continue_train"] = "True"
    training.train_config["pretrained_path"] = str(latest)
    training.logger.info(f"Ablation auto-resume source: {latest}")
else:
    training.train_config["continue_train"] = "False"


class AblationMainInfo(training.MainInfo):
    """Resume from the epoch after the saved checkpoint."""

    def load_latest_model(self, net):
        super().load_latest_model(net)
        if training.MODE == "train":
            self.start_epoch += 1


training.MainInfo = AblationMainInfo


if __name__ == "__main__":
    training.main()
