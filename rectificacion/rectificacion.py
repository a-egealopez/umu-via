#!/usr/bin/env python
"""
Rectificación de perspectiva para medir distancias reales en un plano.

Uso:
    python rectificacion.py --refs=coins_refs.txt [--units=mm] [--dev=imagen.png]

Controles:
    LClick  : añadir punto de medición (2 clics = segmento + distancia)
    RClick  : limpiar medición actual
    r       : activar/desactivar ventana rectificada
"""

import sys
import argparse
import cv2 as cv
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import CAMERA_URL, UI_THEME
from utils import ensure_dev_arg
from ui import draw_hud, draw_circle, draw_line, draw_text
from umucv.stream import autoStream

# ── Argumentos ────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(add_help=False)
ap.add_argument('--refs',  default='refs.txt',
                help='Archivo de referencias: pixel_x pixel_y real_x real_y')
ap.add_argument('--units', default='mm', help='Unidades de las coordenadas reales')
args, _ = ap.parse_known_args()

ensure_dev_arg(CAMERA_URL)

# ── Carga de referencias ──────────────────────────────────────────────────────
REFS_PATH = Path(__file__).parent / args.refs

def load_refs(path):
    img_pts, real_pts, labels = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            img_pts.append([float(parts[0]), float(parts[1])])
            real_pts.append([float(parts[2]), float(parts[3])])
            labels.append(' '.join(parts[4:]) if len(parts) > 4 else '')
    return np.array(img_pts, np.float32), np.array(real_pts, np.float32), labels

try:
    pts_img, pts_real, labels = load_refs(REFS_PATH)
except FileNotFoundError:
    print(f"\nERROR: No se encuentra '{REFS_PATH}'.")
    print("Usa primero pick_refs.py para seleccionar puntos de referencia.")
    sys.exit(1)

if len(pts_img) < 4:
    print(f"\nERROR: Se necesitan al menos 4 referencias (hay {len(pts_img)}).")
    sys.exit(1)

# ── Homografía imagen → plano real ────────────────────────────────────────────
H, hmask = cv.findHomography(pts_img, pts_real, cv.RANSAC, 3.0)
if H is None:
    print("\nERROR: No se pudo calcular la homografía. Revisa los puntos de referencia.")
    sys.exit(1)

n_inliers = int(hmask.sum()) if hmask is not None else len(pts_img)
print(f"Homografía calculada: {n_inliers}/{len(pts_img)} inliers")

def px_to_real(x, y):
    """Transforma un punto imagen (px) a coordenadas reales."""
    r = cv.perspectiveTransform(np.array([[[x, y]]], np.float32), H)
    return r[0, 0]

# ── Vista rectificada ─────────────────────────────────────────────────────────
def build_rectified_transform(frame):
    """Calcula la transformación que elimina la perspectiva y ajusta al canvas."""
    mn = pts_real.min(axis=0)
    mx = pts_real.max(axis=0)
    span = mx - mn
    if span[0] < 1e-6 or span[1] < 1e-6:
        return None, None, None, None

    OUT_W, OUT_H, MARGIN = 640, 480, 35
    s = min((OUT_W - 2*MARGIN) / span[0], (OUT_H - 2*MARGIN) / span[1])

    # M convierte coords reales → píxeles del canvas rectificado
    M = np.array([[s, 0, MARGIN - mn[0]*s],
                  [0, s, MARGIN - mn[1]*s],
                  [0, 0, 1]], np.float64)

    # H_disp = M @ H  →  imagen original → canvas rectificado
    H_disp = M @ H.astype(np.float64)
    return H_disp, s, mn, (OUT_W, OUT_H)

def make_rectified(frame, H_disp, size):
    rect = cv.warpPerspective(frame, H_disp, size)
    return rect

def real_to_rect_px(rx, ry, s, mn, margin=35):
    """Convierte coords reales → píxeles en imagen rectificada."""
    return (int((rx - mn[0]) * s + margin),
            int((ry - mn[1]) * s + margin))

# ── Estado de medición ────────────────────────────────────────────────────────
T = UI_THEME
clicks = []         # [(px, py), ...]  máx. 2
show_rect = [False]

def mouse_cb(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        if len(clicks) >= 2:
            clicks.clear()
        clicks.append((x, y))
    elif event == cv.EVENT_RBUTTONDOWN:
        clicks.clear()

# ── Dibujo ────────────────────────────────────────────────────────────────────
def draw_refs_overlay(frame):
    """Marca los puntos de referencia en la imagen original."""
    for i, (px, py) in enumerate(pts_img):
        px, py = int(px), int(py)
        cv.drawMarker(frame, (px, py), T['C_ACCENT'],
                      cv.MARKER_CROSS, 14, 2, cv.LINE_AA)
        cv.putText(frame, str(i + 1),
                   (px + 6, py - 6), T['F_FONT'], 0.38,
                   T['C_ACCENT'], T['F_THICK'], T['F_LTYPE'])

def draw_measurement(frame):
    """Dibuja los puntos de clic y la medición sobre la imagen original."""
    for px, py in clicks:
        cv.circle(frame, (px, py), 7, T['C_FILL'], -1, cv.LINE_AA)
        cv.circle(frame, (px, py), 7, T['C_TEXT'],  1, cv.LINE_AA)

    if len(clicks) == 2:
        (x1, y1), (x2, y2) = clicks
        r1 = px_to_real(x1, y1)
        r2 = px_to_real(x2, y2)
        dist = np.linalg.norm(r1 - r2)

        cv.line(frame, (x1, y1), (x2, y2), T['C_LINE'], 2, cv.LINE_AA)
        mid = (int((x1 + x2) / 2) + 8, int((y1 + y2) / 2) - 10)
        label = f'{dist:.1f} {args.units}'
        cv.putText(frame, label, mid, T['F_FONT'], 0.65,
                   T['C_TEXT'], T['F_THICK'], T['F_LTYPE'])

def draw_measurement_on_rect(rect_img, s, mn):
    """Proyecta la medición sobre la imagen rectificada."""
    pts_rect = []
    for px, py in clicks:
        rx, ry = px_to_real(px, py)
        pts_rect.append(real_to_rect_px(rx, ry, s, mn))

    for p in pts_rect:
        cv.circle(rect_img, p, 7, T['C_FILL'], -1, cv.LINE_AA)
        cv.circle(rect_img, p, 7, T['C_TEXT'],  1, cv.LINE_AA)

    if len(pts_rect) == 2:
        cv.line(rect_img, pts_rect[0], pts_rect[1], T['C_LINE'], 2, cv.LINE_AA)

# ── Main ──────────────────────────────────────────────────────────────────────
cv.namedWindow('Rectificacion')
cv.setMouseCallback('Rectificacion', mouse_cb)

H_disp = None   # se calcula con el primer frame

for key, frame in autoStream():
    if key == ord('r'):
        show_rect[0] = not show_rect[0]

    # Calcular transformación rectificada (solo una vez, asumiendo cámara fija)
    if H_disp is None:
        H_disp, s_rect, mn_rect, rect_size = build_rectified_transform(frame)

    display = frame.copy()
    draw_refs_overlay(display)
    draw_measurement(display)

    # HUD informativo
    dist_str = '---'
    if len(clicks) == 2:
        r1 = px_to_real(*clicks[0])
        r2 = px_to_real(*clicks[1])
        dist_str = f'{np.linalg.norm(r1 - r2):.1f} {args.units}'

    mode = 'RECT:ON [r]' if show_rect[0] else '[r]:rectif'
    info = (f'Refs:{len(pts_img)} | Dist:{dist_str} | '
            f'LClick:medir  RClick:reset | {mode}')
    cv.imshow('Rectificacion', draw_hud(display, info))

    # Ventana rectificada (toggle con 'r')
    if show_rect[0] and H_disp is not None:
        rect_img = make_rectified(frame, H_disp, rect_size)
        draw_measurement_on_rect(rect_img, s_rect, mn_rect)
        hud_rect = (f'Vista rectificada | escala={s_rect:.2f} px/{args.units} '
                    f'| Dist:{dist_str}')
        cv.imshow('Rectificada', draw_hud(rect_img, hud_rect))
    else:
        try:
            cv.destroyWindow('Rectificada')
        except Exception:
            pass

cv.destroyAllWindows()
