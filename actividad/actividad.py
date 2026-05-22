#!/usr/bin/env python

from __future__ import annotations
import os, sys
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Thread, Lock
from queue import Queue

import cv2 as cv
import face_recognition
from dotenv import load_dotenv
from telegram.ext import Updater

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config import CAMERA_URL, EVENTS_DIR, PRE_SEC, POST_SEC, MIN_AREA_MOV
from utils import ensure_dev_arg, save_temp_image, remove_if_exists, save_frames_to_video, clean_mask, apply_roi_mask, FPSTracker
from umucv.stream import autoStream
from ui import draw_hud
from umucv.util import putText

ensure_dev_arg(CAMERA_URL)

# ── Constantes ────────────────────────────────────────────────────────────────
COLORES              = {"persona": (0, 0, 255), "objeto": (255, 0, 0)}
FACE_UPDATE_INTERVAL = 8
FACE_SCALE           = 0.25
WORK_WIDTH           = 640
WORK_HEIGHT          = 480
_DEFAULT_HISTORY     = 300
_DEFAULT_THRESHOLD   = 50
_MORPH_KERNEL        = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))


# ── Telegram ──────────────────────────────────────────────────────────────────
def init_telegram():
    load_dotenv(Path(__file__).parent / "token.env")
    token, user_id = os.environ["TOKEN"], os.environ["USER_ID"]
    return Updater(token).bot, user_id

_tg_bot, _TG_USER_ID = init_telegram()


def send_telegram_photo(frame, cats):
    tmp = str(EVENTS_DIR / "_telegram_capture.jpg")
    save_temp_image(frame, tmp)
    caption = f"Evento: {', '.join(set(cats))}\n{datetime.now():%d/%m/%Y %H:%M:%S}"
    try:
        with open(tmp, "rb") as f:
            _tg_bot.send_photo(chat_id=_TG_USER_ID, photo=f, caption=caption)
    finally:
        remove_if_exists(tmp)


# ── Cara ──────────────────────────────────────────────────────────────────────
def detect_faces(frame, model):
    inv   = 1 / FACE_SCALE
    small = cv.resize(frame, (0, 0), fx=FACE_SCALE, fy=FACE_SCALE)
    locations = face_recognition.face_locations(
        cv.cvtColor(small, cv.COLOR_BGR2RGB),
        model=model
    )
    return [(int(t*inv), int(r*inv), int(b*inv), int(l*inv)) for t, r, b, l in locations]


def anonymize(frame, face_state, face_queue):
    if face_state["frame_count"] % FACE_UPDATE_INTERVAL == 0 and not face_queue.full():
        face_queue.put((frame.copy(), face_state["detector"]))
    face_state["frame_count"] += 1

    out = frame.copy()
    h, w = frame.shape[:2]
    with face_state["lock"]:
        faces = list(face_state["last_faces"])
    for t, r, b, l in faces:
        dx, dy = int((r - l) * 0.35), int((b - t) * 0.35)
        out[max(0, t-dy):min(h, b+dy), max(0, l-dx):min(w, r+dx)] = 0
    return out


def face_worker(face_state, face_queue):
    grace_frames = 0
    MAX_GRACE = 3

    while True:
        item = face_queue.get()
        if item is None:
            break
        frame, model = item
        faces = detect_faces(frame, model)
        with face_state["lock"]:
            if faces:
                face_state["last_faces"] = faces
                grace_frames = MAX_GRACE
            elif grace_frames > 0:
                grace_frames -= 1
            else:
                face_state["last_faces"] = []


# ── Clasificación ─────────────────────────────────────────────────────────────
def classify_motion(frame, mask_roi, min_area=MIN_AREA_MOV):
    results = []
    for cnt in cv.findContours(mask_roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv.boundingRect(cnt)
        ratio = h / max(w, 1)
        fill  = area / max(w * h, 1)
        cat   = "persona" if ratio > 1.35 and h > 60 and fill < 0.8 else "objeto"
        results.append((cat, x, y, w, h))
    return results


# ── Grabación de eventos ──────────────────────────────────────────────────────
class EventRecorder:
    def __init__(self, pre_sec, post_sec):
        self.pre_sec   = pre_sec
        self.post_sec  = post_sec
        self.pre_buf   = deque(maxlen=max(1, int(pre_sec * 25)))
        self.frames    = []
        self.post_cnt  = 0
        self.recording = False

    def update_pre_buf_size(self, fps):
        self.pre_buf = deque(self.pre_buf, maxlen=max(1, int(self.pre_sec * fps)))

    def push_pre(self, frame):
        self.pre_buf.append(frame)

    def start(self, frame, cats, fps, telegram_enabled):
        self.recording = True
        self.frames    = list(self.pre_buf)
        self.post_cnt  = max(1, int(self.post_sec * fps))
        if telegram_enabled:
            send_telegram_photo(frame, cats)

    def feed(self, frame, active, fps):
        self.frames.append(frame.copy())
        if active:
            self.post_cnt = max(1, int(self.post_sec * fps))
        else:
            self.post_cnt -= 1

    @property
    def finished(self):
        return self.recording and self.post_cnt <= 0

    def save(self, fps, telegram_enabled):
        if telegram_enabled and self.frames:
            path = EVENTS_DIR / f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            save_frames_to_video(self.frames, path, fps)
        self.recording = False
        self.frames    = []
        self.post_cnt  = 0


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    sub = cv.createBackgroundSubtractorMOG2(
        history=_DEFAULT_HISTORY, varThreshold=_DEFAULT_THRESHOLD, detectShadows=True
    )
    recorder  = EventRecorder(PRE_SEC, POST_SEC)
    fps_track = FPSTracker()

    face_state = {"last_faces": [], "frame_count": 0, "detector": "hog", "lock": Lock()}
    roi        = {"pts": [], "selected": False}
    telegram_enabled = False
    detection_mode   = "persona"

    face_queue = Queue(maxsize=1)
    Thread(target=face_worker, args=(face_state, face_queue), daemon=True).start()

    def mouse_callback(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and len(roi["pts"]) < 2:
            roi["pts"].append((x, y))
            if len(roi["pts"]) == 2:
                roi["selected"] = True

    cv.namedWindow("Actividad")
    cv.setMouseCallback("Actividad", mouse_callback)

    frame_idx = 0
    last_out  = None

    for key, frame in autoStream():
        frame_idx += 1
        fps = fps_track.tick()
        recorder.update_pre_buf_size(fps)

        frame = cv.resize(frame, (WORK_WIDTH, WORK_HEIGHT))

        if key == ord("r"):
            sub = cv.createBackgroundSubtractorMOG2(
                history=_DEFAULT_HISTORY, varThreshold=_DEFAULT_THRESHOLD, detectShadows=True
            )
        elif key == ord("t"):
            telegram_enabled = not telegram_enabled
        elif key == ord("m"):
            detection_mode = "objeto" if detection_mode == "persona" else "persona"
        elif key == ord("h"):
            face_state["detector"] = "hog" if face_state["detector"] == "cnn" else "cnn"

        # Frames pares: reutilizar last_out ya anonimizado
        if frame_idx % 2 == 0:
            recorder.push_pre(last_out if last_out is not None else anonymize(frame, face_state, face_queue))
            if last_out is not None:
                cv.imshow("Actividad", last_out)
            continue

        out = anonymize(frame, face_state, face_queue)
        fg  = clean_mask(sub.apply(frame), _MORPH_KERNEL)
        motion_cats = []

        if roi["selected"]:
            mask = apply_roi_mask(fg, roi["pts"])
            (x1, y1), (x2, y2) = roi["pts"]
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])

            for cat, x, y, w, h in classify_motion(frame, mask):
                if not (xmin <= x <= xmax and ymin <= y <= ymax):
                    continue
                motion_cats.append(cat)
                color = COLORES[cat]
                cv.rectangle(out, (x, y), (x + w, y + h), color, 2)
                cv.putText(out, cat.upper(), (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv.rectangle(out, roi["pts"][0], roi["pts"][1], (0, 255, 255), 2)

        face_detected = bool(face_state["last_faces"])
        evento = detection_mode in motion_cats or (detection_mode == "persona" and face_detected)

        if evento and not recorder.recording:
            recorder.start(out, motion_cats, fps, telegram_enabled)

        if recorder.recording:
            recorder.feed(out, evento, fps)
            if recorder.finished:
                recorder.save(fps, telegram_enabled)

        recorder.push_pre(out)

        info = (
            f"FPS={fps:.1f} "
            f"TELEGRAM={'ON' if telegram_enabled else 'OFF'} "
            f"MODO={detection_mode.upper()} "
            f"REC={'ON' if recorder.recording else 'OFF'} "
            f"METODO DE DETECCION={face_state['detector'].upper()}"
        )

        hud = draw_hud(out, info)
        putText(hud, "r=reset | t=telegram | m=objeto/persona | h=hog/cnn", orig=(10, 30))
        last_out = hud
        cv.imshow("Actividad", hud)

    face_queue.put(None)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()