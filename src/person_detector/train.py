"""Fine-tune a YOLOv8 detector on the person-only CrowdHuman dataset."""
from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from .utils import find_best_weights


def finetune(
    data_yaml: str,
    base_model: str = "yolov8n.pt",
    project: str = "./runs/person",
    name: str = "oneclass_stage1_crowdhuman",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cuda",
    patience: int = 15,
    lr0: float = 0.01,
    optimizer: str = "auto",
    cache: str | None = None,
    seed: int = 42,
    resume_if_done: bool = True,
) -> Path:
    """Fine-tune from COCO-pretrained weights and return the best checkpoint.

    The public YOLOv8 weights already know the person class, so the head only has to
    be narrowed to one class instead of learning the concept from scratch.

    With optimizer="auto", Ultralytics picks AdamW with its own learning rate on runs
    under 10k iterations and ignores lr0. A short second fine-tune that needs a low lr0
    has to name the optimizer.
    """
    project = str(Path(project).resolve())

    # Reuse a finished run unless the caller asked for a fresh one.
    if resume_if_done:
        try:
            best = find_best_weights(Path(project) / name)
            print(f"Found {best}, skipping training (use --force-retrain to redo).")
            return best
        except FileNotFoundError:
            pass

    model = YOLO(base_model)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        lr0=lr0,
        optimizer=optimizer,
        cache=cache or False,
        seed=seed,
        project=project,
        name=name,
        exist_ok=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Best checkpoint: {best}")
    return best.resolve()
