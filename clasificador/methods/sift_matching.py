#!/usr/bin/env python

from __future__ import annotations
import cv2 as cv
import numpy as np

# ── Constantes ────────────────────────────────────────────────────────────────
_LOWE_RATIO = 0.75


# ── Factoría ──────────────────────────────────────────────────────────────────
def create() -> SIFTMethod:
    return SIFTMethod(cv.SIFT_create(), cv.BFMatcher(cv.NORM_L2, crossCheck=False))


# ── Clase principal ───────────────────────────────────────────────────────────
class SIFTMethod:
    name = "sift"

    def __init__(self, sift, matcher) -> None:
        self._sift    = sift
        self._matcher = matcher

    def precompute(self, img_bgr: np.ndarray) -> np.ndarray | None:
        _, descs = self._sift.detectAndCompute(cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY), None)
        return descs

    def compare(self, frame_bgr: np.ndarray, descriptor: np.ndarray | None) -> float:
        if descriptor is None:
            return 0.0
        _, descs = self._sift.detectAndCompute(cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY), None)
        if descs is None or len(descs) == 0:
            return 0.0
        good = [
            m for m, n in self._matcher.knnMatch(descs, descriptor, k=2)
            if m.distance < _LOWE_RATIO * n.distance
        ]
        return len(good) / max(min(len(descs), len(descriptor)), 1)

    def best_match(self, frame_bgr: np.ndarray, descriptors: list) -> tuple[int, list[float]]:
        sims = [self.compare(frame_bgr, d) for d in descriptors]
        return int(np.argmax(sims)), sims