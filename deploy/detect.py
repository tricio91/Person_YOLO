"""Person detector service for the NUC: camera in, detections out.

Runs the exported model on the CPU with OpenVINO or ONNX Runtime, without Ultralytics or PyTorch,
so the Docker image stays small. It detects every class the model has (person, and dog
with the two-class model), but only people raise an alert: every processed frame with
people in it is logged as one JSON line on stdout, dogs included, and a snapshot with
the boxes is saved at most once per --cooldown seconds. A frame with only the dog raises
nothing.

    python detect.py --runtime <openvino|onnx> --model <exported model> \\
        --source rtsp://user:pass@192.168.1.50:554/stream

--source also takes a video file, a folder of images or a USB camera index ("0").
Every option can be set through an environment variable of the same name in capitals
(RUNTIME, MODEL, SOURCE, ...), which is how docker-compose passes them.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PAD_VALUE = 114  # the grey Ultralytics pads with during training
CLASS_NAMES = ("person", "dog")  # class order of the models; the one-class model has only person
COLORS = {"person": (0, 0, 255), "dog": (255, 160, 0)}  # BGR: people red, dog blue


class OpenVINOModel:
    def __init__(self, path: str):
        import openvino as ov

        p = Path(path)
        xml = p if p.suffix == ".xml" else next(p.glob("*.xml"))
        self.compiled = ov.Core().compile_model(xml, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
        self.request = self.compiled.create_infer_request()
        self.input_hw = tuple(int(d) for d in self.compiled.input(0).shape[2:])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.request.infer({0: x})[self.compiled.output(0)]


class ONNXModel:
    def __init__(self, path: str):
        import onnxruntime as ort

        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_hw = tuple(int(d) for d in inp.shape[2:])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x})[0]


def letterbox(img: np.ndarray, hw: tuple[int, int]):
    """Resize keeping the aspect ratio and pad with grey up to hw, as in training."""
    h, w = img.shape[:2]
    r = min(hw[0] / h, hw[1] / w)
    nh, nw = round(h * r), round(w * r)
    dh, dw = (hw[0] - nh) / 2, (hw[1] - nw) / 2
    top, left = round(dh - 0.1), round(dw - 0.1)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, hw[0] - nh - top, left, hw[1] - nw - left,
                             cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    x = img[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255  # BGR->RGB, NCHW
    return np.ascontiguousarray(x), r, left, top


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float, max_det: int = 300) -> list[int]:
    """Greedy NMS on float xyxy boxes, same result as Ultralytics.

    cv2.dnn.NMSBoxes was tried first and kept ~4% fewer boxes on crowded images.
    """
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size and len(keep) < max_det:
        i, rest = order[0], order[1:]
        keep.append(i)
        w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = w * h
        order = rest[inter / (area[i] + area[rest] - inter) <= iou]
    return keep


def postprocess(out: np.ndarray, r: float, left: int, top: int, shape, conf: float, iou: float):
    """YOLOv8 head (1, 4+classes, anchors) -> {class name: [[x1, y1, x2, y2, score], ...]}.

    Boxes are in image pixels. Works with the one-class model (person) and the two-class
    one (person, dog). As in Ultralytics, each anchor takes its highest-scoring class,
    and NMS runs per class, so a dog never comes out as a person just because its person
    score passes `conf`.
    """
    pred = out[0].T  # (anchors, 4 + classes)
    names = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class{c}" for c in range(pred.shape[1] - 4)]
    cls = pred[:, 4:]
    best, scores = cls.argmax(axis=1), cls.max(axis=1)
    keep = scores > conf
    pred, best, scores = pred[keep], best[keep], scores[keep]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2] / 2, pred[:, 3] / 2
    boxes = np.stack([cx - bw, cy - bh, cx + bw, cy + bh], axis=1)
    h, w = shape[:2]
    dets = {name: [] for name in names}
    for c, name in enumerate(names):
        idx = np.flatnonzero(best == c)
        for i in (idx[k] for k in nms(boxes[idx], scores[idx], iou)):
            x1, y1, x2, y2 = boxes[i]
            x1, x2 = np.clip([(x1 - left) / r, (x2 - left) / r], 0, w)
            y1, y2 = np.clip([(y1 - top) / r, (y2 - top) / r], 0, h)
            dets[name].append([round(float(v), 1) for v in (x1, y1, x2, y2)] + [round(float(scores[i]), 3)])
    return dets


class LatestFrame:
    """Keep reading a live stream in the background and hand out only the newest frame.

    Without it, a detector slower than the camera falls behind and works on frames
    that are seconds old.
    """

    def __init__(self, source):
        self.source = source
        self.frame = None
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                log(event="source_unavailable", source=str(self.source))
                time.sleep(5)
                continue
            while True:
                ok, frame = cap.read()
                if not ok:
                    log(event="stream_lost", source=str(self.source))
                    break
                with self.lock:
                    self.frame = frame
            cap.release()
            time.sleep(2)

    def read(self):
        with self.lock:
            frame, self.frame = self.frame, None
        return frame


def frames(source: str, fps: float):
    """Yield (name, frame). Files are read in full; live sources are sampled at fps."""
    p = Path(source)
    if p.is_dir():
        for f in sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXTS):
            yield f.name, cv2.imread(str(f))
        return
    if p.is_file() and p.suffix.lower() in IMG_EXTS:
        yield p.name, cv2.imread(str(p))
        return
    if p.is_file():  # video file
        cap = cv2.VideoCapture(str(p))
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            n += 1
            yield f"{p.stem}_{n:06d}", frame
    live = LatestFrame(int(source) if source.isdigit() else source)
    period = 1 / fps if fps > 0 else 0
    while True:
        t0 = time.monotonic()
        frame = live.read()
        if frame is not None:
            yield datetime.now().strftime("%Y%m%d-%H%M%S"), frame
        time.sleep(max(0.0, period - (time.monotonic() - t0)) or 0.01)


def draw(img, dets):
    for name, boxes in dets.items():
        color = COLORS.get(name, (0, 255, 0))
        for x1, y1, x2, y2, s in boxes:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, f"{name} {s:.2f}", (int(x1), max(15, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


def log(**fields):
    print(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **fields}), flush=True)


def env(name, default):
    return os.environ.get(name.upper(), default)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runtime", choices=["openvino", "onnx"], default=env("runtime", "") or None,
                    required=not env("runtime", ""))
    ap.add_argument("--model", default=env("model", ""),
                    help="default: /models/best_openvino_model or /models/best.onnx")
    ap.add_argument("--source", default=env("source", ""), required=not env("source", ""))
    ap.add_argument("--conf", type=float, default=float(env("conf", 0.5)))
    ap.add_argument("--iou", type=float, default=float(env("iou", 0.7)))
    ap.add_argument("--min-height", type=float, default=float(env("min_height", 0.05)),
                    help="drop person boxes shorter than this fraction of the frame height (0 = off)")
    ap.add_argument("--fps", type=float, default=float(env("fps", 5)),
                    help="frames per second taken from a live source")
    ap.add_argument("--save-dir", default=env("save_dir", "/events"))
    ap.add_argument("--cooldown", type=float, default=float(env("cooldown", 10)),
                    help="minimum seconds between two saved snapshots")
    args = ap.parse_args()
    # Anything that is not a URL or a camera index must be a local file or folder.
    # Otherwise a typo in a path would be retried forever as if it were a stream.
    if "://" not in args.source and not args.source.isdigit() and not Path(args.source).exists():
        ap.error(f"source not found: {args.source}")

    default_model = {"openvino": "/models/best_openvino_model", "onnx": "/models/best.onnx"}
    model_path = args.model or default_model[args.runtime]
    model = (OpenVINOModel if args.runtime == "openvino" else ONNXModel)(model_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log(event="start", runtime=args.runtime, model=model_path,
        input=list(model.input_hw), source=args.source)

    last_save, n, total_ms, t_stats = 0.0, 0, 0.0, time.monotonic()
    for name, img in frames(args.source, args.fps):
        if img is None:
            continue
        t0 = time.perf_counter()
        x, r, left, top = letterbox(img, model.input_hw)
        dets = postprocess(model(x), r, left, top, img.shape, args.conf, args.iou)
        # Relative to the frame, so it holds whatever resolution the camera streams at.
        # Only people are filtered: the dog is small and never raises an alert anyway.
        people = [d for d in dets.get("person", []) if d[3] - d[1] >= args.min_height * img.shape[0]]
        dets["person"] = people
        dogs = dets.get("dog", [])
        ms = (time.perf_counter() - t0) * 1000
        n, total_ms = n + 1, total_ms + ms
        if people:  # only people raise an alert; a frame with just the dog is ignored
            log(event="people", frame=name, count=len(people), boxes=people,
                dogs=dogs, ms=round(ms, 1))
            if time.monotonic() - last_save >= args.cooldown:
                cv2.imwrite(str(save_dir / f"{Path(name).stem}_{len(people)}p.jpg"), draw(img.copy(), dets))
                last_save = time.monotonic()
        if time.monotonic() - t_stats >= 60:
            log(event="stats", frames=n, mean_ms=round(total_ms / n, 1))
            n, total_ms, t_stats = 0, 0.0, time.monotonic()


if __name__ == "__main__":
    main()
