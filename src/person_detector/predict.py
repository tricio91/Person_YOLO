"""Run the detector on your own images or video and save the annotated output."""
from __future__ import annotations

from pathlib import Path

import cv2
from ultralytics import YOLO

from .utils import IMG_EXTS


def predict(
    weights: str,
    source: str,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = "cpu",
    project: str = "./runs/person",
    name: str = "predict",
    save_txt: bool = False,
) -> Path:
    """Detect people in `source` and save the drawn boxes under project/name.

    Args:
        weights: checkpoint to run (.pt, .onnx or an *_openvino_model folder).
        source: an image, a folder of images or a video file.
        imgsz: inference input size; should match the training imgsz.
        conf: minimum confidence for a box to be kept.
        device: where to run, e.g. "cpu" or "cuda".
        project: parent folder for the output.
        name: output subfolder, overwritten on every run.
        save_txt: also write one YOLO-format .txt per image with the boxes.
    """
    model = YOLO(weights)
    # stream=True yields one result at a time; without it a long video is
    # held in memory until the end.
    results = model.predict(
        source=source, imgsz=imgsz, conf=conf, device=device,
        save=True, save_txt=save_txt, save_conf=save_txt,
        project=str(Path(project).resolve()), name=name, exist_ok=True,
        stream=True, verbose=False,
    )
    frames = people = empty = 0
    for r in results:
        n = len(r.boxes)
        frames += 1
        people += n
        empty += n == 0
    save_dir = Path(model.predictor.save_dir)
    print(f"{frames} frames, {people} people detected, {empty} frames with nobody")
    print(f"Output: {save_dir}")
    return save_dir


def compare_with_base(
    weights: str,
    base_weights: str,
    source: str,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = "cpu",
    out_dir: str = "./runs/person/compare",
) -> Path:
    """Save each image with the fine-tuned and the base COCO detections side by side.

    The base model knows 80 classes, so it is restricted to class 0 (person).
    Images only: a video would produce one comparison file per frame.
    """
    src = Path(source)
    if src.is_dir():
        paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    elif src.suffix.lower() in IMG_EXTS:
        paths = [src]
    else:
        raise ValueError(f"compare needs an image or a folder of images, got {source}")
    if not paths:
        raise FileNotFoundError(f"No images found in {source}")

    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tuned, base = YOLO(weights), YOLO(base_weights)
    args = dict(imgsz=imgsz, conf=conf, device=device, verbose=False)

    counts = []
    for p in paths:
        rt = tuned.predict(str(p), **args)[0]
        rb = base.predict(str(p), classes=[0], **args)[0]
        left = _label(rt.plot(), f"fine-tuned: {len(rt.boxes)}")
        right = _label(rb.plot(), f"COCO base: {len(rb.boxes)}")
        cv2.imwrite(str(out / f"{p.stem}.jpg"), cv2.hconcat([left, right]))
        counts.append((p.name, len(rt.boxes), len(rb.boxes)))

    differ = [c for c in counts if c[1] != c[2]]
    print(f"{len(counts)} images | people: fine-tuned {sum(c[1] for c in counts)}, "
          f"base {sum(c[2] for c in counts)} | {len(differ)} images with different counts")
    for name, t, b in differ:
        print(f"  {name}: fine-tuned {t}, base {b}")
    print(f"Output: {out}")
    return out


def _label(img, text: str):
    """Write a caption on a black band above the image."""
    band = max(30, img.shape[0] // 20)
    img = cv2.copyMakeBorder(img, band, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    scale = band / 40
    cv2.putText(img, text, (10, int(band * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), max(1, int(scale * 2)), cv2.LINE_AA)
    return img
