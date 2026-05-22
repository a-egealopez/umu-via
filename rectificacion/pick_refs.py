#!/usr/bin/env python

# python pick_refs.py --out=mis_refs.txt [--dev=imagen.png]

import sys
import cv2 as cv
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import UI_THEME
from ui import draw_hud
from umucv.stream import autoStream

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument('--out', default='refs_template.txt')
args, _ = ap.parse_known_args()

OUT_PATH = Path(__file__).parent / args.out
T = UI_THEME

# (pixel_x, pixel_y, real_x, real_y)
points = []
pending = None   # (px, py) esperando input del usuario


def mouse_cb(event, x, y, flags, param):
    global pending
    if event == cv.EVENT_LBUTTONDOWN:
        pending = (x, y)
    elif event == cv.EVENT_RBUTTONDOWN:
        if points:
            p = points.pop()
            print(f'  Eliminado punto {len(points)+1}: pixel ({p[0]}, {p[1]}) → real ({p[2]:.1f}, {p[3]:.1f})')


def ask_real_coords(px, py):
    print(f'\n  Punto {len(points)+1} — pixel ({px}, {py})')
    print('  Introduce coordenadas reales (x y) — primer punto suele ser 0 0:')
    while True:
        try:
            raw = input('  > ').strip()
            rx, ry = map(float, raw.split())
            return rx, ry
        except (ValueError, EOFError):
            print('  Formato incorrecto. Escribe dos números separados por espacio, ej: 0 0')


def save_refs():
    with open(OUT_PATH, 'w') as f:
        f.write('# pixel_x  pixel_y  real_x  real_y\n')
        for px, py, rx, ry in points:
            f.write(f'{px:.1f}  {py:.1f}  {rx:.1f}  {ry:.1f}\n')
    print(f'\n  Guardado en: {OUT_PATH}')
    print(f'  Ejecuta: python rectificacion.py --refs={OUT_PATH.name}')


cv.namedWindow('pick_refs')
cv.setMouseCallback('pick_refs', mouse_cb)

print('\nInstrucciones:')
print('  LClick : anadir punto (te pedirá coords reales en consola)')
print('  RClick : eliminar último punto')
print('  S      : guardar en', OUT_PATH)
print('  C      : limpiar todos')
print('  Q/Esc  : salir')
print()

for key, frame in autoStream():
    base_frame = frame.copy()
    break

while True:
    # Procesar click pendiente (input bloquea, la ventana se congela brevemente)
    if pending is not None:
        px, py = pending
        pending = None
        rx, ry = ask_real_coords(px, py)
        points.append((px, py, rx, ry))
        print(f'  ✓ ({px}, {py}) → real ({rx:.1f}, {ry:.1f})')

    key = cv.waitKey(20) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('s'):
        if len(points) >= 4:
            save_refs()
        else:
            print(f'  Necesitas al menos 4 puntos (tienes {len(points)})')
    elif key == ord('c'):
        points.clear()
        print('  Puntos borrados.')

    display = base_frame.copy()

    for i, (px, py, rx, ry) in enumerate(points):
        cv.drawMarker(display, (px, py), T['C_ACCENT'],
                      cv.MARKER_CROSS, 14, 2, cv.LINE_AA)
        cv.putText(display, f'{i+1} ({rx:.0f},{ry:.0f})', (px + 6, py - 6),
                   T['F_FONT'], 0.38, T['C_ACCENT'], T['F_THICK'], T['F_LTYPE'])

    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv.line(display, points[i][:2], points[i+1][:2],
                    T['C_TRACK'], 1, cv.LINE_AA)
        if len(points) >= 4:
            cv.line(display, points[-1][:2], points[0][:2],
                    T['C_TRACK'], 1, cv.LINE_AA)

    n = len(points)
    estado = 'OK — S para guardar' if n >= 4 else f'Faltan {4-n} puntos mínimo'
    info = f'Puntos: {n} | {estado} | LClick:anadir  RClick:quitar  C:limpiar  S:guardar  Q:salir'
    cv.imshow('pick_refs', draw_hud(display, info))

cv.destroyAllWindows()
