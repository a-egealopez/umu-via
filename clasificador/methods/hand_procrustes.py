#!/usr/bin/env python

from __future__ import annotations
import cv2 as cv
import numpy as np
import mediapipe as mp
from scipy.spatial import procrustes

# ── Configuración ─────────────────────────────────────────────────────────────
_MP_HANDS = mp.solutions.hands
_HANDS_CFG = dict(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1,
)
_DISPARITY_SCALE = 10.0


# ── Factoría ──────────────────────────────────────────────────────────────────
def create() -> HandGestureMethod:
    return HandGestureMethod(_MP_HANDS.Hands(**_HANDS_CFG))


# ── Helpers ───────────────────────────────────────────────────────────────────
def _extract_landmarks(detector, img_bgr: np.ndarray) -> np.ndarray | None:
    rgb = cv.cvtColor(cv.flip(img_bgr, 1), cv.COLOR_BGR2RGB)
    results = detector.process(rgb)
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0]
    return np.array([[l.x, l.y] for l in lm.landmark])


def _normalize_shape(pts: np.ndarray) -> np.ndarray:
    pts = pts - pts.mean(axis=0)
    norm = np.linalg.norm(pts)
    return pts / norm if norm > 0 else pts


# ── Clase principal ───────────────────────────────────────────────────────────
class HandGestureMethod:
    name = "hands"

    def __init__(self, detector) -> None:
        self._detector = detector

    def precompute(self, img_bgr: np.ndarray) -> np.ndarray | None:
        pts = _extract_landmarks(self._detector, img_bgr)
        return _normalize_shape(pts) if pts is not None else None

    def compare(self, frame_bgr: np.ndarray, descriptor: np.ndarray | None) -> float:
        if descriptor is None:
            return 0.0
        pts = _extract_landmarks(self._detector, frame_bgr)
        if pts is None:
            return 0.0
        _, _, disparity = procrustes(descriptor, _normalize_shape(pts))
        return 1.0 / (1.0 + disparity * _DISPARITY_SCALE)

    def best_match(self, frame_bgr: np.ndarray, descriptors: list) -> tuple[int, list[float]]:
        sims = [self.compare(frame_bgr, d) for d in descriptors]
        return int(np.argmax(sims)), sims