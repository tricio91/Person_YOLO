#!/usr/bin/env bash
# End-to-end run: prepare -> train -> validate -> export -> benchmark.
# Point DATA_ROOT at the folder holding images/{train,val} and the official
# annotation_{train,val}.odgt from crowdhuman.org.
set -euo pipefail

DATA_ROOT="${1:?usage: run_pipeline.sh <crowdhuman_root>}"
DATA_YAML="$DATA_ROOT/data.yaml"
WEIGHTS="runs/person/oneclass_stage1_crowdhuman/weights/best.pt"

person-detect prepare --data-root "$DATA_ROOT"
person-detect train   --data "$DATA_YAML" --model yolov8n.pt --epochs 100 --patience 20
person-detect val     --weights "$WEIGHTS" --data "$DATA_YAML"
person-detect export  --weights "$WEIGHTS" --format openvino
person-detect bench   --weights "$WEIGHTS" --images "$DATA_ROOT/images/val" --device cpu
person-detect bench   --weights runs/person/oneclass_stage1_crowdhuman/weights/best_openvino_model --images "$DATA_ROOT/images/val" --device cpu
