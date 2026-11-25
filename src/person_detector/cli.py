"""Command-line entry point wiring the pipeline stages together.

    person-detect prepare  --data-root ~/datasets/crowdhuman
    person-detect train    --data ~/datasets/crowdhuman/data.yaml
    person-detect val      --weights runs/person/oneclass_stage1_crowdhuman/weights/best.pt --data ...
    person-detect export   --weights ... --format openvino
    person-detect bench    --weights ... --images ...
    person-detect predict  --weights ... --source ...
"""
from __future__ import annotations

import argparse

from . import evaluate, export, odgt, predict, train
from .utils import describe_device, pick_device, seed_everything


def _add_prepare(sub):
    """Register `prepare`: .odgt annotations to person-only YOLO labels."""
    p = sub.add_parser("prepare", help="convert CrowdHuman .odgt to person-only YOLO labels")
    p.add_argument("--data-root", required=True,
                   help="folder holding images/{train,val} and annotation_{train,val}.odgt")
    p.add_argument("--box", choices=["fbox", "vbox", "hbox"], default="vbox",
                   help="vbox = visible part only (default), fbox = full body, hbox = head")
    p.add_argument("--min-size", type=float, default=2.0,
                   help="drop boxes thinner than this (px) once clipped to the frame")
    p.add_argument("--workers", type=int, default=8,
                   help="threads used to read the image dimensions")


def _add_train(sub):
    """Register `train`: fine-tune YOLOv8 on the prepared dataset."""
    p = sub.add_parser("train", help="fine-tune YOLOv8 on the person dataset")
    p.add_argument("--data", required=True, help="path to data.yaml from `prepare`")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--project", default="./runs/person")
    p.add_argument("--name", default="oneclass_stage1_crowdhuman",
                   help="run folder inside --project; a new name keeps earlier runs intact")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--optimizer", choices=["auto", "SGD", "AdamW", "Adam"], default="auto",
                   help="auto lets Ultralytics pick one and ignores --lr0 on short runs")
    p.add_argument("--cache", choices=["ram", "disk"], default=None,
                   help="keep the decoded images in memory or on disk between epochs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-retrain", action="store_true")


def _add_val(sub):
    """Register `val`: report mAP, precision and recall for a checkpoint."""
    p = sub.add_parser("val", help="validate a checkpoint")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--imgsz", type=int, default=640)


def _add_export(sub):
    """Register `export`: convert a checkpoint to ONNX or OpenVINO."""
    p = sub.add_parser("export", help="export a checkpoint to onnx/openvino")
    p.add_argument("--weights", required=True)
    p.add_argument("--format", choices=["onnx", "openvino"], required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--dynamic", action="store_true")
    p.add_argument("--half", action="store_true",
                   help="OpenVINO FP16 IR (no calibration needed)")
    p.add_argument("--int8", action="store_true",
                   help="OpenVINO INT8 quantization; requires --data for calibration")
    p.add_argument("--data", default=None,
                   help="data.yaml used to draw INT8 calibration images from")


def _add_bench(sub):
    """Register `bench`: time inference for any weights format."""
    p = sub.add_parser("bench", help="benchmark inference latency")
    p.add_argument("--weights", required=True)
    p.add_argument("--images", required=True, help="folder of images to time on")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--out-json", default=None)


def _add_predict(sub):
    """Register `predict`: draw detections on your own images or video."""
    p = sub.add_parser("predict", help="detect people in your own images or video")
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="image, folder of images or video")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--project", default="./runs/person")
    p.add_argument("--name", default="predict")
    p.add_argument("--save-txt", action="store_true",
                   help="also write the boxes as YOLO .txt files")
    p.add_argument("--compare-base", nargs="?", const="yolov8n.pt", default=None,
                   metavar="WEIGHTS",
                   help="also run the COCO model (default yolov8n.pt) and save both side by side")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with every subcommand attached."""
    parser = argparse.ArgumentParser(
        prog="person-detect",
        description="YOLOv8 person detector: CrowdHuman -> fine-tune -> ONNX/OpenVINO.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for add in (_add_prepare, _add_train, _add_val, _add_export, _add_bench, _add_predict):
        add(sub)
    return parser


def main() -> None:
    """Parse the arguments and run the matching pipeline stage."""
    args = build_parser().parse_args()

    # prepare touches no model, so it skips the device probe below.
    if args.command == "prepare":
        odgt.prepare(
            data_root=args.data_root,
            box=args.box,
            min_size=args.min_size,
            workers=args.workers,
        )
        return

    device = pick_device()
    describe_device(device)

    if args.command == "train":
        seed_everything(args.seed)
        train.finetune(
            data_yaml=args.data,
            base_model=args.model,
            project=args.project,
            name=args.name,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            patience=args.patience,
            lr0=args.lr0,
            optimizer=args.optimizer,
            cache=args.cache,
            seed=args.seed,
            resume_if_done=not args.force_retrain,
        )
    elif args.command == "val":
        evaluate.validate(args.weights, args.data, imgsz=args.imgsz, device=device)
    elif args.command == "export":
        if args.format == "onnx":
            export.to_onnx(args.weights, imgsz=args.imgsz, opset=args.opset, dynamic=args.dynamic)
        else:
            export.to_openvino(
                args.weights, imgsz=args.imgsz,
                half=args.half, int8=args.int8, data=args.data,
            )
    elif args.command == "bench":
        evaluate.benchmark(
            args.weights, args.images, imgsz=args.imgsz,
            device=args.device, limit=args.limit, out_json=args.out_json,
        )
    elif args.command == "predict":
        predict.predict(
            args.weights, args.source, imgsz=args.imgsz, conf=args.conf, device=device,
            project=args.project, name=args.name, save_txt=args.save_txt,
        )
        if args.compare_base:
            predict.compare_with_base(
                args.weights, args.compare_base, args.source,
                imgsz=args.imgsz, conf=args.conf, device=device,
                out_dir=f"{args.project}/{args.name}_compare",
            )


if __name__ == "__main__":
    main()
