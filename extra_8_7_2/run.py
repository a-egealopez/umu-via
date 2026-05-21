#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import cv2 as cv

_ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from config import CAMERA_URL
from utils import ensure_dev_arg
from umucv.stream import autoStream
from umucv.util import parser, parse

from hand_controller import HandController
from ar_viewer import ARViewer
from ui import draw_hud

if len(sys.argv) == 1:
    ensure_dev_arg(CAMERA_URL)

parser.add_argument("--model", default="", metavar="RUTA",
                    help="Archivo .obj generado por COLMAP/VGGT en Colab (opcional)")
args = parse()

WIN = "AR Gestual"
cv.namedWindow(WIN)

hand = HandController()
ar   = ARViewer()

if args.model:
    obj_path = Path(args.model)
    if obj_path.exists():
        ar.load(obj_path)
    else:
        print(f"[WARN] No se encontró '{obj_path}' — usando cubo de referencia.")
        ar.use_fallback()
else:
    ar.use_fallback()

cv.setMouseCallback(WIN, ar.mouse_cb)

stream = autoStream()
print("Controles: clic izq=anclar | r=reset | q=salir")

for key, frame in stream:
    if key == ord("r"):
        ar.reset()

    state = hand.process(frame)
    ar.update(state, frame.shape)
    ar.draw(frame)
    hand.draw_landmarks(frame, state)

    cv.imshow(WIN, draw_hud(frame, ar.status(state)))

hand.close()
cv.destroyAllWindows()
