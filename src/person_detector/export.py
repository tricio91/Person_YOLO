"""Export a trained checkpoint to ONNX or to OpenVINO IR.

The two exports are independent: each one is converted straight from the .pt. OpenVINO
is what the project runs on the Intel NUC; ONNX, which it was benchmarked against (see
the README), is the alternative.
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def to_onnx(
    weights: str,
    imgsz: int = 640,
    opset: int = 12,
    dynamic: bool = False,
    simplify: bool = True,
) -> Path:
    """Export .pt -> .onnx and return the resulting file path.

    Args:
        weights: trained checkpoint (.pt) to export.
        imgsz: fixed input size in pixels; should match the training imgsz.
        opset: ONNX operator-set version; lower is more compatible with older runtimes.
        dynamic: allow variable batch and image size instead of a fixed 1x3xHxW input.
        simplify: fold constants and drop redundant nodes for a leaner graph.
    """
    model = YOLO(weights)
    path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
    )
    print(f"ONNX model: {path}")
    return Path(path)


def to_openvino(
    weights: str,
    imgsz: int = 640,
    half: bool = False,
    int8: bool = False,
    data: str | None = None,
) -> Path:
    """Export .pt to an OpenVINO IR folder (the .xml/.bin pair) and return its path.

    half=True writes an FP16 IR: half the size, no calibration needed. The runtime
    converts it back to FP32 at load unless the CPU handles FP16 natively, so the gain
    is size rather than speed.

    int8=True runs post-training quantization, which needs calibration images taken
    from the training domain. Ultralytics reads them from its `data` argument and
    falls back to coco8.yaml when it is missing, which produces a worse model without
    reporting anything, so `data` is required here. Output lands in a separate
    *_int8_openvino_model folder.
    """
    if int8 and not data:
        raise ValueError(
            "int8=True needs data=<path to data.yaml> for calibration; without it "
            "Ultralytics silently calibrates on coco8.yaml."
        )
    if int8 and half:
        raise ValueError("half and int8 are mutually exclusive; pick one.")

    model = YOLO(weights)
    path = model.export(
        format="openvino", imgsz=imgsz, half=half, int8=int8, data=data
    )
    print(f"OpenVINO model: {path}")
    return Path(path)
