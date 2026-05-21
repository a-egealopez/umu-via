#!/usr/bin/env python
"""
Herramienta interactiva para seleccionar puntos de referencia en una imagen.

Uso:
    python pick_refs.py --out=mis_refs.txt [--dev=imagen.png]

Controles:
    LClick  : añadir punto de referencia (muestra coordenadas pixel en terminal)
    RClick  : eliminar último punto
    S       : guardar plantilla (rellena las columnas real_x real_y después)
    C       : limpiar todos los puntos
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
from ui import draw_hud
from umucv.stream import autoStream

# ── Argumentos ────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(add_help=False)
ap.add_argument('--out', default='refs_template.txt',
                help='Ruta del archivo de salida (plantilla)')
args, _ = ap.parse_known_args()

ensure_dev_arg(CAMERA_URL)

OUT_PATH = Path(__file__).parent / args.out
T = UI_THEME

# ── Estado ────────────────────────────────────────────────────────────────────
points = []   # lista de (x, y)

def mouse_cb(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f'  Punto {len(points):2d}: pixel ({x:4d}, {y:4d})')
    elif event == cv.EVENT_RBUTTONDOWN:
        if points:
            removed = points.pop()
            print(f'  Eliminado punto {len(points)+1}: pixel {removed}')

def save_template():
    """Guarda la plantilla con pixel coords y columnas real_x real_y vacías."""
    with open(OUT_PATH, 'w') as f:
        f.write('# Archivo de referencias para rectificacion.py\n')
        f.write('# Formato: pixel_x  pixel_y  real_x  real_y  [etiqueta]\n')
        f.write('# 1. Sustituye ??? por las coordenadas reales medidas.\n')
        f.write('# 2. Usa las mismas unidades en todas las filas (mm, cm, m...).\n')
        f.write('# 3. Necesitas al menos 4 puntos no colineales.\n')
        f.write('#\n')
        for i, (x, y) in enumerate(points):
            f.write(f'# Punto {i + 1}\n')
            f.write(f'{x:.1f}  {y:.1f}  ???.?  ???.?\n')
    print(f'\nPlantilla guardada en: {OUT_PATH}')
    print('Edita el archivo y rellena los ??? con las coordenadas reales.')
    print('Después ejecuta: python rectificacion.py --refs=' + OUT_PATH.name)

# ── Main ──────────────────────────────────────────────────────────────────────
cv.namedWindow('pick_refs')
cv.setMouseCallback('pick_refs', mouse_cb)

print('\nInstrucciones:')
print('  LClick : añadir punto de referencia')
print('  RClick : eliminar último punto')
print('  S      : guardar plantilla en', OUT_PATH)
print('  C      : limpiar todos los puntos')
print()

for key, frame in autoStream():
    if key == ord('s'):
        if len(points) >= 4:
            save_template()
        else:
            print(f'  Necesitas al menos 4 puntos (tienes {len(points)})')
    elif key == ord('c'):
        points.clear()
        print('  Puntos borrados.')

    display = frame.copy()

    # Dibujar puntos seleccionados
    for i, (x, y) in enumerate(points):
        cv.drawMarker(display, (x, y), T['C_ACCENT'],
                      cv.MARKER_CROSS, 14, 2, cv.LINE_AA)
        cv.putText(display, str(i + 1), (x + 6, y - 6),
                   T['F_FONT'], 0.42, T['C_ACCENT'],
                   T['F_THICK'], T['F_LTYPE'])

    # Unir puntos con líneas para ver el cuadrilátero
    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv.line(display, points[i], points[i + 1],
                    T['C_TRACK'], 1, cv.LINE_AA)
        if len(points) >= 4:
            cv.line(display, points[-1], points[0],
                    T['C_TRACK'], 1, cv.LINE_AA)

    n = len(points)
    estado = 'OK (S para guardar)' if n >= 4 else f'Faltan {4 - n} puntos mínimo'
    info = (f'Puntos: {n} | {estado} | '
            f'LClick:añadir  RClick:quitar  C:limpiar  S:guardar')
    cv.imshow('pick_refs', draw_hud(display, info))

cv.destroyAllWindows()
