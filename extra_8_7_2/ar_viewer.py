#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np


_CUBE_V = np.array([
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
], dtype=np.float32)

_CUBE_E = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def load_obj(path: str | Path, max_edges: int = 4_000) -> tuple[np.ndarray, list[tuple[int, int]]]:
    verts: list[list[float]] = []
    edges: set[tuple[int, int]] = set()

    with open(path) as fh:
        for raw in fh:
            tok = raw.split()
            if not tok:
                continue
            if tok[0] == "v" and len(tok) >= 4:
                verts.append([float(tok[1]), float(tok[2]), float(tok[3])])
            elif tok[0] == "f" and len(tok) >= 4:
                idx = [int(t.split("/")[0]) - 1 for t in tok[1:]]
                for k in range(len(idx)):
                    a, b = idx[k], idx[(k + 1) % len(idx)]
                    edges.add((min(a, b), max(a, b)))

    if not verts:
        raise ValueError(f"No se encontraron vértices en {path}")

    # sin caras -> nube de puntos (ej. COLMAP/VGGT); derivar aristas por convex hull
    if not edges:
        from scipy.spatial import ConvexHull
        arr = np.array(verts, dtype=np.float32)
        try:
            hull = ConvexHull(arr)
            for simplex in hull.simplices:
                for k in range(len(simplex)):
                    a, b = int(simplex[k]), int(simplex[(k + 1) % len(simplex)])
                    edges.add((min(a, b), max(a, b)))
        except Exception as exc:
            raise ValueError(f"Sin caras en {path} y convex hull fallido: {exc}") from exc

    edge_list = sorted(edges)
    if len(edge_list) > max_edges:
        step = max(1, len(edge_list) // max_edges)
        edge_list = edge_list[::step]

    return np.array(verts, dtype=np.float32), edge_list


def _normalize(v: np.ndarray) -> np.ndarray:
    v = v - v.mean(0)
    s = float(np.abs(v).max())
    return v / s if s > 0 else v


def _Rx(a: float) -> np.ndarray:
    c, s = float(np.cos(a)), float(np.sin(a))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def _Ry(a: float) -> np.ndarray:
    c, s = float(np.cos(a)), float(np.sin(a))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


class ARViewer:

    _SCALE_MIN = 0.1
    _SCALE_MAX = 4.5
    _XY_FRAC   = 0.60
    _SMOOTH    = 0.25
    _SMOOTH_RY = 0.18   # roll_deg viene de z-depth, más ruidoso

    def __init__(self) -> None:
        self._verts: np.ndarray = _normalize(_CUBE_V)
        self._edges: list       = list(_CUBE_E)
        self._anchor: list[int] = [320, 240]
        self._scale: float      = 1.0
        self._ry:    float      = 0.0
        self._rx:    float      = 0.0
        self._ox:    float      = 0.0
        self._oy:    float      = 0.0

        self._base_px: int      = 200    # default para el cubo

    def load(self, path: str | Path) -> None:
        v, e = load_obj(path)

        v[1, :] = v[1, :] * -1  # invertimoooos el eje Y debido a los modelos Colmap/Vggt

        self._verts = _normalize(v)
        self._edges = e

        print(f"[AR] {Path(path).name}: {len(v)} verts, {len(e)} aristas")

    def use_fallback(self) -> None:
        self._verts = _normalize(_CUBE_V)
        self._edges = list(_CUBE_E)

        self._base_px = 80 # se cambia para los modelos (distinto del cubo)
        
        print("[AR] Cubo de referencia (sin modelo .obj)")

    def reset(self) -> None:
        self._scale = 1.0
        self._ry = self._rx = self._ox = self._oy = 0.0

    def mouse_cb(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv.EVENT_LBUTTONDOWN:
            self._anchor = [x, y]

    def update(self, state, frame_shape: tuple) -> None:
        if not state.detected:
            return
        h, w = frame_shape[:2]
        a = self._SMOOTH

        target_scale = self._SCALE_MIN + state.distance * (self._SCALE_MAX - self._SCALE_MIN)
        target_ry    = float(np.radians(state.angle_deg))
        target_rx    = float(np.radians(-state.roll_deg))
        target_ox    = (state.norm_x - 0.5) * w * self._XY_FRAC
        target_oy    = (state.norm_y - 0.5) * h * self._XY_FRAC

        self._scale += a               * (target_scale - self._scale)
        self._ry    += a               * (target_ry    - self._ry)
        self._rx    += self._SMOOTH_RY * (target_rx    - self._rx)
        self._ox    += a               * (target_ox    - self._ox)
        self._oy    += a               * (target_oy    - self._oy)

    def draw(self, frame: np.ndarray) -> None:
        R    = _Ry(self._ry) @ _Rx(self._rx)
        rot  = (R @ self._verts.T).T
        px   = self._base_px * self._scale
        cx   = self._anchor[0] + self._ox
        cy   = self._anchor[1] + self._oy
        
        pts  = rot[:, :2].copy()
        pts[:, 0] = pts[:, 0] * px + cx
        pts[:, 1] = pts[:, 1] * px + cy
        ipts = pts.astype(np.int32)

        if self._edges:
            edges_arr = np.array(self._edges, dtype=np.int32)
            
            lines = ipts[edges_arr]
            
            cv.polylines(frame, lines, isClosed=False, color=(0, 230, 150), thickness=1, lineType=cv.LINE_AA)

        cv.drawMarker(frame,
                      (int(cx), int(cy)),
                      (0, 220, 220), cv.MARKER_CROSS, 14, 1, cv.LINE_AA)

    def status(self, state) -> str:
        if state.detected:
            return (f"escala={self._scale:.2f}  "
                    f"rotacion={np.degrees(self._ry):+.0f}deg  "
                    f"roll={np.degrees(self._rx):+.0f}deg  "
                    f"clic=anclar  r=reset  q=salir")
        return "Sin mano — clic=anclar  r=reset  q=salir"
