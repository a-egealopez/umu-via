#!/usr/bin/env python

from __future__ import annotations
import sys, importlib
from pathlib import Path

import cv2 as cv

_ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from config import CAMERA_URL
from utils import ensure_dev_arg
from umucv.stream import autoStream
from umucv.util import putText, parser, parse
from ui import draw_hud

# ── Argumentos ────────────────────────────────────────────────────────────────
if len(sys.argv) == 1:
    print(f"Uso: python clasificador.py [--method {'|'.join(_METHOD_MAP)}] [--models CARPETA]")
    sys.exit(0)

if len(sys.argv) == 3:
    ensure_dev_arg(CAMERA_URL)

# ── Métodos disponibles ───────────────────────────────────────────────────────
_METHOD_MAP = {
    "embedding": "methods.mp_embedding",
    "hands":     "methods.hand_procrustes",
    "sift":      "methods.sift_matching",
}

# ── Argumentos obligatorios ───────────────────────────────────────────────────
parser.add_argument(
    "--models",
    type=str,
    required=True,
    help="Carpeta donde están los modelos"
)

parser.add_argument(
    "--method",
    type=str,
    required=True,
    choices=_METHOD_MAP.keys(),
    help=f"Método a usar: {' | '.join(_METHOD_MAP)}"
)

args = parse()


EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


# ── Método ────────────────────────────────────────────────────────────────────
def load_method(name):
    if name not in _METHOD_MAP:
        sys.exit(f"[ERROR] Método '{name}' no reconocido. Opciones: {list(_METHOD_MAP)}")
    m = importlib.import_module(_METHOD_MAP[name]).create()
    print(f"[INFO] Método cargado: {m.name}")
    return m


# ── Gestión de modelos ────────────────────────────────────────────────────────
class ModelStore:
    def __init__(self, method, folder):
        self.m       = method
        self.f       = Path(folder)
        self.names   = []
        self.descs   = []
        self.session = []

    def load_from_disk(self):
        self.names, self.descs = [], []
        for f in sorted(f for ext in EXTS for f in self.f.glob(ext)):
            img = cv.imread(str(f))
            if img is None:
                continue
            self.names.append(f.stem)
            self.descs.append(self.m.precompute(img))
            print(f"  [modelo] {f.stem}")
        print(f"[INFO] {len(self.names)} modelos cargados.")

    def add(self, frame, name):
        self.f.mkdir(parents=True, exist_ok=True)
        fpath = self.f / f"{name}.png"
        cv.imwrite(str(fpath), frame)
        self.names.append(name)
        self.descs.append(self.m.precompute(frame))
        self.session.append((name, fpath))
        print(f"[INFO] Guardado: {fpath}")

    def remove_last(self):
        if not self.session:
            return
        name, fpath = self.session.pop()
        if name in self.names:
            idx = self.names.index(name)
            self.names.pop(idx)
            self.descs.pop(idx)
        if fpath.exists():
            fpath.unlink()
        print(f"[INFO] Eliminado: {fpath}")


# ── Dibujo ────────────────────────────────────────────────────────────────────
def draw_results(display, names, result):
    h, w = display.shape[:2]
    putText(display, f"{result['name']} ({result['sim']:.2f})", (10, 20))
    cv.rectangle(display, (0, h - 10), (int(result['sim'] * w), h), (0, 200, 0), -1)
    for i, (nm, sim) in enumerate(zip(names, result['sims'])):
        color = (0, 255, 0) if nm == result['name'] else (180, 180, 180)
        putText(display, f"{nm}: {sim:.2f}", (10, h - 15 - i * 16), color)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    method = load_method(args.method)
    store  = ModelStore(method, args.models)
    store.load_from_disk()

    frame_count = 0
    result = {"name": "?", "sim": 0.0, "sims": []}

    for key, frame in autoStream():
        frame_count += 1

        if key == ord("m"):
            name = input("Nombre del modelo: ").strip() or f"modelo_{len(store.names):03d}"
            store.add(frame, name)
        elif key == ord("d"):
            store.remove_last()

        display = frame.copy()

        if store.names:
            if frame_count % 30 == 0:
                idx, sims = store.m.best_match(frame, store.descs)
                result = {"name": store.names[idx], "sim": float(sims[idx]), "sims": [float(s) for s in sims]}
            draw_results(display, store.names, result)
        else:
            putText(display, "Sin modelos. Pulsa 'm' para capturar uno.")

        hud = draw_hud(display, f"METODO={args.method.upper()} | MODELOS={args.models.upper()} | m=guardar | d=borrar")
        cv.imshow("Clasificador", hud)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()