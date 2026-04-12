from __future__ import annotations
import sys, os, threading
from pathlib import Path

import cv2 as cv
import numpy as np


# ── Stream ────────────────────────────────────────────────────────────────────
def ensure_dev_arg(url: str) -> None:
    if not any(a.startswith("--dev") for a in sys.argv[1:]):
        sys.argv.append(f"--dev={url}")


# ── Máscara ───────────────────────────────────────────────────────────────────
def clean_mask(fg: np.ndarray, kernel: np.ndarray | None = None, shadow_thresh: int = 200) -> np.ndarray:
    if kernel is None:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    _, fg = cv.threshold(fg, shadow_thresh, 255, cv.THRESH_BINARY)
    fg = cv.morphologyEx(fg, cv.MORPH_OPEN,  kernel)
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, kernel)
    return fg


def apply_roi_mask(fg: np.ndarray, roi_pts: list) -> np.ndarray:
    mask = np.zeros_like(fg)
    if len(roi_pts) == 2:
        (x1, y1), (x2, y2) = roi_pts
        mask[y1:y2, x1:x2] = fg[y1:y2, x1:x2]
    return mask


# ── Proyección 3D ─────────────────────────────────────────────────────────────
def project_points(points_3d: np.ndarray, K: np.ndarray,
                   dist=None, rvec=None, tvec=None) -> np.ndarray:
    pts, _ = cv.projectPoints(
        np.asarray(points_3d, dtype=np.float32),
        rvec if rvec is not None else np.zeros(3),
        tvec if tvec is not None else np.zeros(3),
        K,
        dist if dist is not None else np.zeros(5),
    )
    return pts.reshape(-1, 2)


# ── Vídeo ─────────────────────────────────────────────────────────────────────
def save_frames_to_video(frames: list[np.ndarray], path: str | Path, fps: float = 25.0) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


# ── Frame buffer ──────────────────────────────────────────────────────────────
class FrameBuffer:
    def __init__(self) -> None:
        self._frame = None
        self._lock  = threading.Lock()

    def write(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame.copy()

    def read(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


# ── Imagen temporal ───────────────────────────────────────────────────────────
def save_temp_image(frame: np.ndarray, path: str | Path) -> None:
    cv.imwrite(str(path), frame)


def remove_if_exists(path: str | Path) -> None:
    try:
        os.remove(str(path))
    except FileNotFoundError:
        pass


# ── FPS ───────────────────────────────────────────────────────────────────────
class FPSTracker:
    def __init__(self, target: float = 25.0) -> None:
        self._fps   = target
        self._count = 0
        self._t0    = cv.getTickCount()

    def tick(self) -> float:
        self._count += 1
        elapsed = (cv.getTickCount() - self._t0) / cv.getTickFrequency()
        if elapsed >= 1.0:
            self._fps   = self._count / elapsed
            self._count = 0
            self._t0    = cv.getTickCount()
        return self._fps