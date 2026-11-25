"""Helpers shared across the pipeline: device pick, seeding, weight lookup."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pick_device() -> str:
    """Return 'cuda' when a GPU is visible, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_device(device: str) -> None:
    """Print the device in use, plus the GPU name when running on CUDA."""
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")


def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch so a training run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_best_weights(run_dir) -> Path:
    """Find the best.pt Ultralytics writes under a run directory, or raise."""
    run_dir = Path(run_dir)
    candidates = [
        run_dir / "oneclass_stage1_crowdhuman" / "weights" / "best.pt",
        run_dir / "weights" / "best.pt",
        run_dir,  # allow passing a .pt path straight through
    ]
    for p in candidates:
        if p.suffix == ".pt" and p.exists():
            return p.resolve()
        best = p / "best.pt" if p.is_dir() else None
        if best and best.exists():
            return best.resolve()
    raise FileNotFoundError(f"best.pt not found near {run_dir}")
