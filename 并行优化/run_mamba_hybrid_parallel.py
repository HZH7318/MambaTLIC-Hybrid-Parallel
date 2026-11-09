"""Training entry point for the hybrid parallel-tail Mamba codec.

The existing training implementation is reused without modification.  Its
model and loss globals are replaced before ``main`` is called, so existing
configuration files remain compatible (new experiments must start from a
checkpoint produced by this architecture).
"""

import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
MODEL_ROOT = EXPERIMENT_ROOT / "model"

# Make the independent experiment runnable from any current working directory.
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MODEL_ROOT))

from mambamodel_hybrid_parallel import HNet, RateDistortion

import run_mamba_1 as training


training.model = HNet
training.Loss = RateDistortion


class HybridMainInfo(training.MainInfo):
    """Use the epoch after the saved epoch when resuming a training run."""

    def load_latest_model(self, net):
        super().load_latest_model(net)
        if training.MODE == "train":
            self.start_epoch += 1


training.MainInfo = HybridMainInfo


if __name__ == "__main__":
    training.main()
