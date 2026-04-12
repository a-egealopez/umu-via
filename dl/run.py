#!/usr/bin/env python

from __future__ import annotations
import sys, threading
from pathlib import Path

import cv2 as cv

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config import CAMERA_URL
from utils import ensure_dev_arg, FrameBuffer
from umucv.stream import autoStream
from ui import draw_hud
from umucv.util import putText
from ultralytics import YOLO

ensure_dev_arg(CAMERA_URL)

# ── Constantes ────────────────────────────────────────────────────────────────
MODEL_PATH  = Path(__file__).parent / "best.pt"
INFER_SIZE  = 320
CONF_THRESH = 0.3
COLORS      = {"book": (255, 0, 0), "fruit": (0, 255, 0), "toy": (0, 0, 255)}


# ── Hilo de inferencia ────────────────────────────────────────────────────────
class YOLOInferenceThread(threading.Thread):
    def __init__(self, model: YOLO, buf: FrameBuffer) -> None:
        super().__init__(daemon=True)
        self.model   = model
        self.buf     = buf
        self.lock    = threading.Lock()
        self._boxes  = []
        self.running = True

    @property
    def boxes(self) -> list:
        with self.lock:
            return list(self._boxes)

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        while self.running:
            frame = self.buf.read()
            if frame is None:
                continue
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            [res] = self.model(rgb, imgsz=INFER_SIZE, conf=CONF_THRESH)
            with self.lock:
                self._boxes = res.boxes


# ── Dibujo ────────────────────────────────────────────────────────────────────
def draw_boxes(frame, boxes, model: YOLO):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls   = model.names[int(box.cls)]
        conf  = float(box.conf)
        color = COLORS.get(cls, (255, 255, 255))
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv.putText(frame, f"{cls} {conf:.0%}", (x1, y1 - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    model.overrides["verbose"] = False

    buf    = FrameBuffer()
    thread = YOLOInferenceThread(model, buf)
    thread.start()

    for key, frame in autoStream():
        buf.write(frame)
        draw_boxes(frame, thread.boxes, model)
        cv.imshow("YOLO Detector", frame)

    thread.stop()
    thread.join(timeout=2.0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()