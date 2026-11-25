"""Fine-tuned checkpoint vs the stock COCO yolov8n on CrowdHuman val.

Writes what the README shows into docs/comparison/: metrics.json, pr_curve.png,
metrics.png and a side-by-side jpg for every image passed with --images.

    python scripts/compare_models.py --data ~/datasets/crowdhuman/data.yaml \
        --images ~/datasets/crowdhuman/images/val/<name>.jpg ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

from person_detector.predict import compare_with_base

TUNED_COLOR, BASE_COLOR = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f2328", "#59636e", "#e4e7eb"


def evaluate(weights: str, data: str, imgsz: int, device: str, out: Path) -> dict:
    """Validate one model and keep the numbers plus its PR curve."""
    m = YOLO(weights).val(
        data=data, imgsz=imgsz, device=device, verbose=False, plots=False,
        project=str(out / "val"), name=Path(weights).stem, exist_ok=True,
    )
    # curves_results[0] is the PR curve: recall grid and one precision row per class.
    recall, precision = m.box.curves_results[0][:2]
    return {
        "mAP50": float(m.box.map50),
        "mAP50-95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "pr_curve": {"recall": recall.tolist(), "precision": precision[0].tolist()},
    }


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_pr(tuned: dict, base: dict, path: Path):
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
    for res, label, color in ((tuned, "fine-tuned", TUNED_COLOR), (base, "COCO yolov8n", BASE_COLOR)):
        c = res["pr_curve"]
        ax.plot(c["recall"], c["precision"], color=color, linewidth=2,
                label=f"{label}  (AP50 {res['mAP50']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall", color=INK)
    ax.set_ylabel("Precision", color=INK)
    ax.set_title("Precision-recall on CrowdHuman val (IoU 0.5)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_metrics(tuned: dict, base: dict, path: Path):
    keys = ["mAP50", "mAP50-95", "precision", "recall"]
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=150)
    h = 0.36
    for i, k in enumerate(keys):
        for off, res, color in ((-h / 2, tuned, TUNED_COLOR), (h / 2, base, BASE_COLOR)):
            ax.barh(i + off, res[k], height=h - 0.04, color=color)
            ax.text(res[k] + 0.01, i + off, f"{res[k]:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(keys)), keys)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_title("Same val set, same thresholds", color=INK, fontsize=11, loc="left")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=TUNED_COLOR),
                       plt.Rectangle((0, 0), 1, 1, color=BASE_COLOR)],
              labels=["fine-tuned", "COCO yolov8n"], frameon=False, fontsize=9,
              loc="lower right")
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default="runs/person/oneclass_stage1_crowdhuman/weights/best.pt")
    ap.add_argument("--base", default="yolov8n.pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--images", nargs="*", default=[], help="val images to draw side by side")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="docs/comparison")
    ap.add_argument("--max-width", type=int, default=1600,
                    help="downscale the side-by-side images so the repo stays small")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path("runs/person/compare")

    tuned = evaluate(args.weights, args.data, args.imgsz, args.device, scratch)
    base = evaluate(args.base, args.data, args.imgsz, args.device, scratch)
    plot_pr(tuned, base, out / "pr_curve.png")
    plot_metrics(tuned, base, out / "metrics.png")
    summary = {name: {k: round(v, 4) for k, v in res.items() if k != "pr_curve"}
               for name, res in (("fine-tuned", tuned), ("coco_yolov8n", base))}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # CrowdHuman names carry commas, so the outputs are renumbered for the README.
    for i, img in enumerate(args.images, 1):
        compare_with_base(args.weights, args.base, img, imgsz=args.imgsz,
                          device=args.device, out_dir=str(out))
        src = out / f"{Path(img).stem}.jpg"
        im = cv2.imread(str(src))
        src.unlink()
        if im.shape[1] > args.max_width:
            s = args.max_width / im.shape[1]
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"example_{i}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
