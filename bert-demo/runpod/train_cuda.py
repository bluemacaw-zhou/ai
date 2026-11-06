"""Run the existing trainer with mandatory CUDA and per-run persistent outputs."""
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from trainer import Trainer
from training_config import TrainingConfig


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to start CPU training on this GPU Pod")
    output = Path(os.environ.get("OUTPUT_DIR", "/workspace/bert-demo/runs/default")).resolve()
    # Use a fresh run directory, so historical checkpoints cannot be mistaken for this run.
    output.mkdir(parents=True, exist_ok=True)
    for filename in ("best_classifier.pth", "last_classifier.pth"):
        if (output / filename).exists():
            raise FileExistsError(f"Existing checkpoint: {output / filename}; use a new OUTPUT_DIR")
    config = replace(TrainingConfig.parse_arguments(), checkpoint_directory=output)
    print(f"GPU={torch.cuda.get_device_name(0)}, checkpoint_directory={output}")
    Trainer(config).run()


if __name__ == "__main__":
    main()
