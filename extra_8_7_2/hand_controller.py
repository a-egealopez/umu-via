#!/usr/bin/env python
from __future__ import annotations

import os
import logging
import warnings
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np
import mediapipe as mp

os.environ["GLOG_minloglevel"]      = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["GRPC_VERBOSITY"]        = "ERROR"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("mediapipe").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

_MP_HANDS = mp.solutions.hands


@dataclass
class HandState:
    detected:      bool   = False
    distance:      float  = 0.5
    roll_deg:      float  = 0.0
    angle_deg:     float  = 0.0
    norm_x:        float  = 0.5
    norm_y:        float  = 0.5
    landmarks_raw: object = field(default=None, repr=False)


class HandController:

    _BBOX_FAR    = 0.10
    _BBOX_CLOSE  = 0.55
    _INFER_SCALE = 0.5

    def __init__(self, min_det: float = 0.6, min_track: float = 0.6) -> None:
        self._detector = _MP_HANDS.Hands(
            model_complexity=0,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_track,
            max_num_hands=1,
        )

    def process(self, frame_bgr: np.ndarray) -> HandState:
        h, w = frame_bgr.shape[:2]
        sw   = max(1, int(w * self._INFER_SCALE))
        sh   = max(1, int(h * self._INFER_SCALE))
        rgb  = cv.cvtColor(cv.resize(frame_bgr, (sw, sh)), cv.COLOR_BGR2RGB)
        res  = self._detector.process(rgb)

        if not res.multi_hand_landmarks:
            return HandState()

        lm  = res.multi_hand_landmarks[0]
        pts = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)
        z   = np.array([p.z        for p in lm.landmark], dtype=np.float32)

        diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        dist = float(np.clip(
            (diag - self._BBOX_FAR) / (self._BBOX_CLOSE - self._BBOX_FAR),
            0.0, 1.0,
        ))

        roll_deg  = float(np.clip((z[5] - z[17]) * 500, -80, 80))

        dv        = pts[9] - pts[0]
        angle_deg = float(np.degrees(np.arctan2(dv[0], -dv[1])))

        palm = pts[[0, 5, 9, 13, 17]].mean(0)
        return HandState(
            detected=True,
            distance=dist,
            roll_deg=roll_deg,
            angle_deg=angle_deg,
            norm_x=float(palm[0]),
            norm_y=float(palm[1]),
            landmarks_raw=lm,
        )

    def draw_landmarks(self, frame_bgr: np.ndarray, state: HandState) -> None:
        if state.landmarks_raw is None:
            return
        mp.solutions.drawing_utils.draw_landmarks(
            frame_bgr,
            state.landmarks_raw,
            _MP_HANDS.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style(),
        )

    def close(self) -> None:
        self._detector.close()
