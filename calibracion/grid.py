#!/usr/bin/env python

import sys
import cv2 as cv
import numpy as np
from pathlib import Path

# ─── Setup ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import CAMERA_URL, CALIB_FILE, FRAME_W, FRAME_H, UI_THEME
from utils import ensure_dev_arg, project_points
from ui import SliderPanel, draw_hud
from umucv.stream import autoStream

ensure_dev_arg(CAMERA_URL)

# ─── Cámara ────────────────────────────────
vals = np.loadtxt(str(CALIB_FILE))
K = vals[:9].reshape(3, 3)
dist = vals[9:]

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

# ─── Geometría ─────────────────────────────────────────
def ray(x, y):
    return np.array([(x - cx)/fx, (y - cy)/fy, 1.0])

def floor_pt(x, y, h):
    r = ray(x, y)
    return None if r[1] <= 1e-6 else r * (h / r[1])

def wall_pt(x, y, z):
    r = ray(x, y)
    return r * (z / r[2])

# ─── Utils ─────────────────────────────────────────────
def params(p):
    return {"w_z": p["Z"]/100, "c_h": p["A"]/100, "cell": p["X"]/100}

def grid(a, b, s):
    return np.arange(np.floor(a/s)*s, np.ceil(b/s)*s+s, s)

# ─── Dibujo ────────────────────────────────────────────
def line3d(img, a, b, K, col, t=1):
    pts = project_points(np.array([a, b], np.float32), K)
    if np.isfinite(pts).all():
        cv.line(img, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)),
                col, t, UI_THEME["F_LTYPE"])
        return pts

def txt(img, s, p, col, sc=0.4):
    cv.putText(img, s, (int(p[0]), int(p[1])),
               UI_THEME["F_FONT"], sc, col,
               UI_THEME["F_THICK"], UI_THEME["F_LTYPE"])

def draw_grid(img, K, w_z, c_h, cell):
    w = img.shape[1]
    lx = -w_z * cx / fx
    rx =  w_z * (w - cx) / fx
    top = c_h - w_z

    xs = grid(lx, rx, cell)

    for xg in xs:
        line3d(img, [xg, c_h, 0.05], [xg, c_h, w_z], K, UI_THEME["C_TRACK"])
        line3d(img, [xg, top, w_z], [xg, c_h, w_z], K, UI_THEME["C_TRACK"])

    for z in np.arange(0.05, w_z+cell, cell):
        lx2 = -z * cx / fx
        rx2 =  z * (w - cx) / fx
        pts = line3d(img, [lx2, c_h, z], [rx2, c_h, z], K, UI_THEME["C_TRACK"])
        if pts is not None:
            txt(img, f"{z*100:.0f}", pts[0]+[5,-5], UI_THEME["C_TEXT"], 0.35)

    for y in np.arange(top, c_h+cell, cell):
        line3d(img, [lx, y, w_z], [rx, y, w_z], K, UI_THEME["C_TRACK"])

    line3d(img, [lx, c_h, w_z], [rx, c_h, w_z], K, UI_THEME["C_LINE"], 2)

# ─── Clicks ────────────────────────────────
points = []

def mouse(event, x, y, flags, param):
    if event != cv.EVENT_LBUTTONDOWN:
        return

    p = params(panel)

    fp = floor_pt(x, y, p["c_h"])
    wp = wall_pt(x, y, p["w_z"])

    candidates = []

    if fp is not None:
        candidates.append((fp, "floor"))

    if wp is not None:
        candidates.append((wp, "wall"))

    if not candidates:
        return

    pt, plane = min(
        candidates,
        key=lambda c: np.linalg.norm(c[0])
    )

    if len(points) == 2:
        points.clear()

    points.append((x, y, pt, plane))

def draw_points(img):
    for x, y, _, plane in points:
        col = UI_THEME["C_VALUE"] if plane == "wall" else UI_THEME["C_FILL"]
        cv.circle(img, (x, y), 5, col, -1)
        txt(img, plane, (x+10, y-10), col)

    if len(points) == 2:
        (x1,y1,p1,_), (x2,y2,p2,_) = points
        cv.line(img, (x1,y1), (x2,y2), UI_THEME["C_LINE"], 2)

        d = np.linalg.norm(p1 - p2) * 100
        mid = ((x1+x2)//2, (y1+y2)//2)
        txt(img, f"{d:.1f} cm", (mid[0]+10, mid[1]-10), UI_THEME["C_TEXT"], 0.6)

# ─── Main ──────────────────────────────────────────────
panel = SliderPanel("Sliders", [
    ("Z", 5, 300, 150, " cm"),
    ("A", 5, 300,  50, " cm"),
    ("X", 5, 200,  50, " cm"),
])

cv.namedWindow("Calibracion")
cv.setMouseCallback("Calibracion", mouse)

w, h = FRAME_W, FRAME_H
new_K, _ = cv.getOptimalNewCameraMatrix(K, dist, (w, h), 1)
map1, map2 = cv.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv.CV_32FC1)

for key, frame in autoStream():
    frame = cv.remap(frame, map1, map2, cv.INTER_LINEAR)

    p = params(panel)
    draw_grid(frame, new_K, **p)
    draw_points(frame)

    info = f"Z={p['w_z']*100:.0f} A={p['c_h']*100:.0f} X={p['cell']*100:.0f} | 2 Clicks=Draw measurement segment"
    cv.imshow("Calibracion", draw_hud(frame, info))

    panel.show()

cv.destroyAllWindows()