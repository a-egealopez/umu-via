#!/usr/bin/env python
"""
Controlador gestual de YouTube — MediaPipe + Procrustes.

  Gestos (Procrustes):   puno=pausa, pulgar_dcha=avanzar, pulgar_izq=retroceder
  Distancia mano-cámara: cerca=acelerar, lejos=decelerar
  Ángulo inclinación:    arriba=subir volumen, abajo=bajar volumen

Uso:
  python gesture.py --dev=<CAMERA_URL>
  python gesture.py --models=./mis_gestos/ --dev=<CAMERA_URL>
"""

from __future__ import annotations

import os, sys, time, warnings, logging
from pathlib import Path

# Silenciar warnings de MediaPipe / TF / absl / protobuf antes de importar nada
os.environ["GLOG_minloglevel"]    = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"]       = "ERROR"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("mediapipe").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import cv2 as cv
import numpy as np
import mediapipe as mp
from scipy.spatial import procrustes

_ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from config import CAMERA_URL
from utils import ensure_dev_arg
from umucv.stream import autoStream
from umucv.util import putText, parser, parse
from ui import draw_hud


# -- Configuración --

_MP_HANDS = mp.solutions.hands
_HANDS_CFG = dict(
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    max_num_hands=1,
)

# Gestos predefinidos y sus teclas YouTube
GESTURES = [
    ("fist",        "k",  "Puno cerrado  -> Pausa/Play (k)"),
    ("thumb_right", "l",  "Pulgar derecha -> Avanzar 10s (l)"),
    ("thumb_left",  "j",  "Pulgar izquierda -> Retroceder 10s (j)"),
    ("open",        None, "Mano abierta  -> Neutro (sin accion)"),
]
GESTURE_KEYS   = {g[0]: g[1] for g in GESTURES}
GESTURE_LABELS = {
    "fist":        "Puno -> Pausa/Play",
    "thumb_right": "Pulgar dcha -> Avanzar",
    "thumb_left":  "Pulgar izq -> Retroceder",
    "open":        "Mano abierta (neutro)",
}

SIMILARITY_THRESHOLD = 0.55
COOLDOWN_GESTURE     = 1.0
COOLDOWN_SPEED       = 1.5
COOLDOWN_VOLUME      = 0.3
DIST_CLOSE           = 0.35
DIST_FAR             = 0.15
ANGLE_UP             = 25.0
ANGLE_DOWN           = -25.0
DISPARITY_SCALE      = 10.0
MATCH_EVERY_N        = 5
MP_SCALE             = 0.5

SPEED_STEPS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

COL_GREEN = (0, 220, 100)
COL_RED   = (60, 60, 255)


# -- Argumentos --

if len(sys.argv) == 1:
    print("Uso: python gesture.py --dev=<URL> [--models=CARPETA]")
    sys.exit(0)

if len(sys.argv) == 3:
    ensure_dev_arg(CAMERA_URL)

parser.add_argument("--models", type=str, default="./gestures",
                    help="Carpeta con imagenes de gestos modelo")
args = parse()


# -- Teclado (pynput) --

def _init_keyboard():
    from pynput.keyboard import Controller, Key

    class Keyboard:
        _special = {"shift": Key.shift, "ctrl": Key.ctrl, "alt": Key.alt,
                     "up": Key.up, "down": Key.down}
        def __init__(self):
            self._kb = Controller()

        def press(self, key: str):
            self._kb.press(key)
            self._kb.release(key)

        def hotkey(self, *keys: str):
            mapped = [self._special.get(k, k) for k in keys]
            for k in mapped:
                self._kb.press(k)
            for k in reversed(mapped):
                self._kb.release(k)

    return Keyboard()


# -- MediaPipe helpers --

def extract_landmarks(detector, img_bgr):
    small = cv.resize(img_bgr, None, fx=MP_SCALE, fy=MP_SCALE)
    rgb = cv.cvtColor(small, cv.COLOR_BGR2RGB)
    results = detector.process(rgb)
    if not results.multi_hand_landmarks:
        return None, None
    lm = results.multi_hand_landmarks[0]
    pts = np.array([[l.x, l.y] for l in lm.landmark], dtype=np.float32)
    return pts, lm


def normalize_shape(pts):
    pts = pts - pts.mean(axis=0)
    n = np.linalg.norm(pts)
    return pts / n if n > 0 else pts


def hand_size(pts):
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def hand_angle(pts):
    d = pts[12] - pts[0]   # muneca -> dedo corazón
    return float(np.degrees(np.arctan2(d[0], -d[1])))


# -- Almacén de gestos --

class GestureStore:
    def __init__(self, detector, folder):
        self.detector = detector
        self.folder   = Path(folder)
        self.names    = []
        self.descs    = []
        self._session = []
        self.folder.mkdir(parents=True, exist_ok=True)

    def load(self):
        self.names.clear()
        self.descs.clear()
        files = sorted(f for ext in EXTS for f in self.folder.glob(ext))
        for f in files:
            img = cv.imread(str(f))
            if img is None:
                continue
            pts, _ = extract_landmarks(self.detector, img)
            if pts is not None:
                self.names.append(f.stem)
                self.descs.append(normalize_shape(pts))
        if self.names:
            print(f"Modelos cargados: {', '.join(self.names)}")

    def add(self, frame, name):
        path = self.folder / f"{name}.png"
        cv.imwrite(str(path), frame)
        pts, _ = extract_landmarks(self.detector, frame)
        if pts is not None:
            self.names.append(name)
            self.descs.append(normalize_shape(pts))
            self._session.append((name, path))
            print(f"  Gesto '{name}' guardado")
        else:
            print(f"  No se detecto mano al guardar '{name}'")

    def remove_last(self):
        if not self._session:
            return
        name, path = self._session.pop()
        if name in self.names:
            i = self.names.index(name)
            self.names.pop(i)
            self.descs.pop(i)
        if path.exists():
            path.unlink()
        print(f"  Eliminado: {name}")


# -- Matching Procrustes --

def match_gesture(query_pts, store):
    if not store.names:
        return None, 0.0, {}
    qn = normalize_shape(query_pts)
    sims = {}
    for name, desc in zip(store.names, store.descs):
        _, _, disp = procrustes(desc, qn)
        sims[name] = 1.0 / (1.0 + disp * DISPARITY_SCALE)
    best = max(sims, key=sims.get)
    s = sims[best]
    return (best if s >= SIMILARITY_THRESHOLD else None), s, sims


# -- Grabación guiada --

def guided_record(store, autostream_gen):
    """Graba los 4 gestos predefinidos de forma guiada."""
    print("\n--- Grabacion de gestos ---")
    for name, _key, desc in GESTURES:
        print(f"\n  Siguiente gesto: {desc}")
        print("  Muestra el gesto y pulsa ESPACIO para capturar (s=saltar, q=cancelar)")
        for key, frame in autostream_gen:
            display = frame.copy()
            putText(display, f"Grabar: {desc}", (10, 20))
            putText(display, "ESPACIO=capturar  s=saltar  q=cancelar", (10, 40))
            cv.imshow("YouTube Gesture Controller", display)
            if key == ord(" "):
                store.add(frame, name)
                break
            elif key == ord("s"):
                print(f"  Saltado: {name}")
                break
            elif key == ord("q") or key == 27:
                print("  Grabacion cancelada")
                return
    print("\nGrabacion completada\n")


# -- Dibujo --

def draw_results(display, gesture_name, similarity, sz, angle,
                 speed, volume, sims_dict, lm_raw):
    h, w = display.shape[:2]

    if lm_raw is not None:
        mp.solutions.drawing_utils.draw_landmarks(
            display, lm_raw, _MP_HANDS.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())

    label = GESTURE_LABELS.get(gesture_name, gesture_name or "---")
    putText(display, f"GESTO: {label}  ({similarity:.2f})", (10, 20))

    bar_w = int((w // 3) * min(similarity, 1.0))
    col = COL_GREEN if similarity >= SIMILARITY_THRESHOLD else COL_RED
    cv.rectangle(display, (10, h - 10), (10 + bar_w, h), col, -1)

    for i, (nm, sim) in enumerate(sorted(sims_dict.items(), key=lambda x: -x[1])):
        c = COL_GREEN if nm == gesture_name else (180, 180, 180)
        putText(display, f"{nm}: {sim:.3f}", (10, h - 20 - i * 16), c)

    putText(display, f"Vel: {speed:.2f}x",  (w - 140, 20))
    putText(display, f"Vol: {volume}%",      (w - 140, 40))
    putText(display, f"Ang: {angle:+.1f}",   (w - 140, 60))
    putText(display, f"Dist: {sz:.2f}",      (w - 140, 80))


# -- Main --

def main():
    keyboard = _init_keyboard()
    detector = _MP_HANDS.Hands(**_HANDS_CFG)
    store    = GestureStore(detector, args.models)
    store.load()

    speed       = 1.0
    volume      = 50
    t_gesture   = 0.0
    t_speed     = 0.0
    t_volume    = 0.0
    last_sent   = None
    cached      = {"name": None, "sim": 0.0, "sims": {}}
    n_frame     = 0

    if not store.names:
        print("No hay modelos. Pulsa 'r' para grabar gestos o pon imagenes en", args.models)

    stream = autoStream()
    print("Controles: r=grabar gestos | d=borrar ultimo | q=salir")

    for key, frame in stream:
        n_frame += 1
        now = time.time()

        # Grabacion guiada
        if key == ord("r"):
            guided_record(store, stream)
            continue
        if key == ord("d"):
            store.remove_last()
            continue

        pts, lm_raw = extract_landmarks(detector, frame)

        g_name  = None
        sim     = 0.0
        sims    = {}
        sz      = 0.0
        angle   = 0.0

        if pts is not None:
            sz    = hand_size(pts)
            angle = hand_angle(pts)

            if store.names:
                # Matching cada N frames para reducir latencia
                if n_frame % MATCH_EVERY_N == 0:
                    g_name, sim, sims = match_gesture(pts, store)
                    cached = {"name": g_name, "sim": sim, "sims": sims}
                else:
                    g_name = cached["name"]
                    sim    = cached["sim"]
                    sims   = cached["sims"]

                # Enviar tecla de gesto
                if (g_name and g_name != "open"
                        and GESTURE_KEYS.get(g_name)
                        and now - t_gesture > COOLDOWN_GESTURE
                        and g_name != last_sent):
                    keyboard.press(GESTURE_KEYS[g_name])
                    t_gesture = now
                    last_sent = g_name

                if g_name in ("open", None):
                    last_sent = None

                # Velocidad por distancia
                if now - t_speed > COOLDOWN_SPEED:
                    idx = SPEED_STEPS.index(speed) if speed in SPEED_STEPS else 3
                    if sz > DIST_CLOSE and idx < len(SPEED_STEPS) - 1:
                        speed = SPEED_STEPS[idx + 1]
                        keyboard.hotkey("shift", ".")
                        t_speed = now
                    elif 0 < sz < DIST_FAR and idx > 0:
                        speed = SPEED_STEPS[idx - 1]
                        keyboard.hotkey("shift", ",")
                        t_speed = now

                # Volumen por ángulo
                if now - t_volume > COOLDOWN_VOLUME:
                    if angle > ANGLE_UP and volume < 100:
                        volume = min(100, volume + 5)
                        keyboard.press("up")
                        t_volume = now
                    elif angle < ANGLE_DOWN and volume > 0:
                        volume = max(0, volume - 5)
                        keyboard.press("down")
                        t_volume = now

        draw_results(frame, g_name, sim, sz, angle, speed, volume, sims, lm_raw)
        hud = draw_hud(frame, f"r=grabar | d=borrar | vel={speed}x | vol={volume}%")
        cv.imshow("YouTube Gesture Controller", hud)

    detector.close()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()