"""Person detection with YOLOv8: prepare CrowdHuman, fine-tune, export to ONNX/OpenVINO."""

import os

# Ultralytics pip-installs whatever it thinks is missing. Loading an .onnx on the CPU
# makes it install the CPU `onnxruntime` on top of `onnxruntime-gpu`, which overwrites
# the GPU build. Has to be set before ultralytics is imported; export
# YOLO_AUTOINSTALL=true to get the old behaviour back.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

__version__ = "0.1.0"
