"""Convert the official CrowdHuman .odgt annotations into person-only YOLO labels.

The .odgt files are line-delimited JSON: one object per image, holding its ID and a
list of gtboxes. Each box carries three rectangles (fbox = full body, vbox = visible
part, hbox = head) plus the flags the annotators set.

Labels are written next to the images, in place. Ultralytics locates them by swapping
"images" for "labels" in the path, so no copying or linking is involved.

Expected layout under data_root:

    images/train/*.jpg      annotation_train.odgt
    images/val/*.jpg        annotation_val.odgt
"""
from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from .utils import IMG_EXTS

SPLITS = ("train", "val")
BOX_KEYS = ("fbox", "vbox", "hbox")


def _image_size(path: Path):
    """Return (width, height) read from the file header, or None if it cannot be read."""
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def _index_sizes(images_dir: Path, workers: int) -> dict[str, tuple[int, int]]:
    """Map every image stem in a folder to its (width, height)."""
    paths = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
    # Header reads are I/O bound, so threads pay off and memory stays flat.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        sizes = pool.map(_image_size, paths)
    return {p.stem: s for p, s in zip(paths, sizes) if s is not None}


def _convert_record(record: dict, width: int, height: int, box_key: str,
                    min_size: float, stats: Counter) -> list[str]:
    """Return the YOLO label lines for one .odgt record, class id 0 for every person."""
    lines: list[str] = []
    for box in record.get("gtboxes", []):
        tag = box.get("tag")
        # "mask" tags mark crowds left unlabelled. They are not people, but they do
        # contain people, so detections there score as false positives. Ultralytics
        # has no ignore-region support, so they are dropped.
        if tag != "person":
            stats["mask" if tag == "mask" else f"tag_{tag}"] += 1
            continue
        # Boxes the annotators themselves did not trust.
        if (box.get("extra") or {}).get("ignore", 0) == 1:
            stats["dropped_ignore"] += 1
            continue

        raw = box.get(box_key)
        if not raw or len(raw) != 4:
            stats["dropped_malformed"] += 1
            continue

        # CrowdHuman boxes are [x, y, w, h] in pixels from the top-left corner, and
        # run past the frame when a body is cut off by the edge. YOLO needs normalized
        # centre/size within [0, 1], so the box is clipped to the image.
        x, y, w, h = (float(v) for v in raw)
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(width), x + w), min(float(height), y + h)
        cw, ch = x2 - x1, y2 - y1
        if cw < min_size or ch < min_size:
            stats["dropped_tiny"] += 1
            continue
        if (x1, y1, x2, y2) != (x, y, x + w, y + h):
            stats["clipped"] += 1

        stats["kept"] += 1
        lines.append(
            f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{cw / width:.6f} {ch / height:.6f}"
        )
    return lines


def _convert_split(root: Path, split: str, box_key: str, min_size: float,
                   workers: int) -> Counter:
    """Write labels/<split>/*.txt from annotation_<split>.odgt and report the counts."""
    odgt = root / f"annotation_{split}.odgt"
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    if not odgt.exists():
        raise FileNotFoundError(f"Missing {odgt}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing {images_dir}")
    labels_dir.mkdir(parents=True, exist_ok=True)

    sizes = _index_sizes(images_dir, workers)
    stats: Counter = Counter()
    annotated: set[str] = set()

    # Streamed one record at a time. Reading the train file whole parses to about
    # 580 MB of Python objects, against 11 MB like this.
    with odgt.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            image_id = record["ID"]
            annotated.add(image_id)
            size = sizes.get(image_id)
            if size is None:
                stats["missing_or_unreadable_image"] += 1
                continue
            width, height = size
            lines = _convert_record(record, width, height, box_key, min_size, stats)
            (labels_dir / f"{image_id}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
            stats["images"] += 1
            if not lines:
                stats["images_without_person"] += 1

    # An image with no label file is silently treated as background by Ultralytics,
    # so it is worth counting rather than ignoring.
    stats["images_without_annotation"] = len(set(sizes) - annotated)
    return stats


def _clear_stale_caches(root: Path) -> list[Path]:
    """Delete Ultralytics *.cache files so training re-reads the labels just written."""
    stale = sorted(root.rglob("*.cache"))
    for path in stale:
        path.unlink()
    return stale


def prepare(data_root: str, box: str = "vbox", min_size: float = 2.0,
            workers: int = 8) -> Path:
    """Write the YOLO labels and data.yaml under data_root, and return the yaml path."""
    if box not in BOX_KEYS:
        raise ValueError(f"box must be one of {BOX_KEYS}, got {box!r}")
    root = Path(data_root).expanduser().resolve()

    for path in _clear_stale_caches(root):
        print(f"Removed stale cache: {path}")

    totals: dict[str, Counter] = {}
    for split in SPLITS:
        print(f"Converting {split} ...")
        totals[split] = _convert_split(root, split, box, min_size, workers)

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        f"path: {root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: person\n"
    )
    (root / "prepare_stats.json").write_text(
        json.dumps(
            {"box": box, "min_size": min_size,
             "splits": {s: dict(c) for s, c in totals.items()}},
            indent=2,
        )
    )

    print(f"\nDataset ready at {root}  (box={box})")
    for split, stats in totals.items():
        images = stats["images"]
        per_image = stats["kept"] / images if images else 0.0
        print(f"  {split}: {images} images, {stats['kept']} person boxes "
              f"({per_image:.2f}/image)")
        print(f"    dropped -> mask regions: {stats['mask']}, "
              f"ignore-flagged: {stats['dropped_ignore']}, "
              f"below {min_size}px: {stats['dropped_tiny']}")
        print(f"    clipped to frame: {stats['clipped']}, "
              f"images with no person: {stats['images_without_person']}, "
              f"unreadable/missing: {stats['missing_or_unreadable_image']}, "
              f"images with no annotation: {stats['images_without_annotation']}")
    print(f"\n  data.yaml: {data_yaml}")
    return data_yaml
