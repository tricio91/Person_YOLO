"""Validation metrics and a latency benchmark.

The benchmark loads anything Ultralytics understands (.pt, .onnx or an OpenVINO
folder), so one command compares the PyTorch model against its exported variants.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .utils import IMG_EXTS


def validate(weights: str, data_yaml: str, imgsz: int = 640, device: str = "cuda") -> dict:
    """Run the val pass and return mAP50, mAP50-95, precision and recall."""
    model = YOLO(weights)
    m = model.val(data=data_yaml, imgsz=imgsz, device=device, verbose=False)
    metrics = {
        "mAP50": float(m.box.map50),
        "mAP50-95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
    }
    print(f"mAP50={metrics['mAP50']:.3f}  mAP50-95={metrics['mAP50-95']:.3f}  "
          f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}")
    return metrics


def benchmark(
    weights: str,
    images_dir: str,
    imgsz: int = 640,
    device: str = "cpu",
    warmup: int = 3,
    limit: int = 100,
    out_json: str | None = None,
) -> dict:
    """Time per-image inference and report mean, median and FPS.

    Defaults to CPU, which is where the ONNX/OpenVINO exports are worth measuring.

    Args:
        weights: model to time (.pt, .onnx or an *_openvino_model folder).
        images_dir: folder searched recursively for test images.
        imgsz: inference input size; should match the export imgsz.
        device: where to run, e.g. "cpu" or "0" for the first GPU.
        warmup: untimed runs first, so loading and graph compilation are not measured.
        limit: maximum number of images to time.
        out_json: optional path to save the result as JSON.
    """
    model = YOLO(weights)
    paths = [p for p in sorted(Path(images_dir).rglob("*")) if p.suffix.lower() in IMG_EXTS]
    paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No images found under {images_dir}")

    # Warm-up keeps lazy loading and graph compilation out of the timings.
    for p in paths[:warmup]:
        model.predict(str(p), imgsz=imgsz, device=device, verbose=False)

    times = []
    for p in paths:
        t0 = time.perf_counter()
        model.predict(str(p), imgsz=imgsz, device=device, verbose=False)
        times.append(time.perf_counter() - t0)

    arr = np.array(times)
    result = {
        "weights": str(weights),
        "device": device,
        "n_images": len(times),
        "mean_ms": float(arr.mean() * 1000),
        "median_ms": float(np.median(arr) * 1000),
        "fps": float(1.0 / arr.mean()),
    }
    print(f"{Path(str(weights)).name} @ {device}: "
          f"mean={result['mean_ms']:.1f} ms  median={result['median_ms']:.1f} ms  "
          f"{result['fps']:.1f} FPS")
    if out_json:
        Path(out_json).write_text(json.dumps(result, indent=2))
    return result
