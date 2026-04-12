#!/usr/bin/env python

from __future__ import annotations

import sys, time, threading
import cv2 as cv
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ui import SliderPanel, draw_circle, draw_hud
from umucv.stream import autoStream

if len(sys.argv) == 1:
    sys.argv.append("--dev=carretera")

WIN = "Trafico"
cv.namedWindow(WIN)
cv.resizeWindow(WIN, 960, 540)

panel = SliderPanel("Controles", [
    ("Fusion px",   0,    40,   5,  "px"),
    ("Area minima", 10, 1000,  40, "px2"),
    ("Close px",    0,    60,  15,  "px"),
])

# ───────────────────────── STATE ─────────────────────────

BIN_S = 60   # segundos por bin temporal

state = {
    "roi":      [], "roi_ok":   False,
    "split":    [], "split_ok": False,
    "line_x": None, "line_ok":  False,
    "mode": "roi",
    "count":     {"izq": 0, "der": 0},
    "history":   [],          # lista de (timestamp, "izq"|"der")
    "start_time": time.time(),
}

# ─────────────────── CENTROID TRACKER ────────────────────

tracks   = {}
next_id  = [0]
MAX_DIST = 100
MAX_AGE  = 5
ALPHA    = 0.4

def update_tracks(objs):
    matched = set()
    result  = {}
    for obj in objs.values():
        cx, cy, bx, by, bw, bh = obj
        best_id, best_d = None, MAX_DIST + 1
        for tid, t in tracks.items():
            d = np.hypot(cx - t["cx"], cy - t["cy"])
            if d < best_d:
                best_d, best_id = d, tid
        if best_id is not None and best_id not in matched:
            p = tracks[best_id]
            result[best_id] = {
                "cx": int(ALPHA*cx + (1-ALPHA)*p["cx"]),
                "cy": int(ALPHA*cy + (1-ALPHA)*p["cy"]),
                "bx": int(ALPHA*bx + (1-ALPHA)*p["bx"]),
                "by": int(ALPHA*by + (1-ALPHA)*p["by"]),
                "bw": int(ALPHA*bw + (1-ALPHA)*p["bw"]),
                "bh": int(ALPHA*bh + (1-ALPHA)*p["bh"]),
                "age": 0,
            }
            matched.add(best_id)
        else:
            result[next_id[0]] = {
                "cx": cx, "cy": cy,
                "bx": bx, "by": by, "bw": bw, "bh": bh,
                "age": 0,
            }
            next_id[0] += 1
    for tid, t in tracks.items():
        if tid not in matched and t["age"] + 1 < MAX_AGE:
            result[tid] = {**t, "age": t["age"] + 1}
    tracks.clear()
    tracks.update(result)

# ─────────────────────── INPUT ───────────────────────────

def mouse(event, x, y, flags, _):
    if event != cv.EVENT_LBUTTONDOWN:
        return
    if state["mode"] == "roi" and len(state["roi"]) < 4:
        state["roi"].append((x, y))
        if len(state["roi"]) == 4:
            state["roi_ok"] = True
    elif state["mode"] == "split" and len(state["split"]) < 2:
        state["split"].append((x, y))
        if len(state["split"]) == 2:
            state["split_ok"] = True
    elif state["mode"] == "count":
        state["line_x"] = x
        state["line_ok"] = True

cv.setMouseCallback(WIN, mouse)

# ─────────────────────── UTILS ───────────────────────────

def separate(mask, k):
    k = max(1, int(k))
    kern = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
    return cv.dilate(cv.erode(mask, kern), kern)

def split_y(cx):
    split = state["split"]
    if len(split) < 2:
        return 0
    (x0, y0), (x1, y1) = split
    if x1 == x0:
        return y0
    return int(y0 + (y1 - y0) * (cx - x0) / (x1 - x0))

def detect(mask, min_area):
    _, _, stats, cent = cv.connectedComponentsWithStats(mask)
    out = {}
    for i, (cx, cy) in enumerate(cent[1:], start=0):
        area = stats[i+1, cv.CC_STAT_AREA]
        if area < min_area:
            continue
        bx = stats[i+1, cv.CC_STAT_LEFT]
        by = stats[i+1, cv.CC_STAT_TOP]
        bw = stats[i+1, cv.CC_STAT_WIDTH]
        bh = stats[i+1, cv.CC_STAT_HEIGHT]
        aspect = bw / bh if bh > 0 else 1
        if aspect < 0.3 or aspect > 8:
            continue
        out[i] = (int(cx), int(cy), bx, by, bw, bh)
    return out

# ─────────────────────── FRAME ───────────────────────────

def process(frame, bgsub, paused):
    vis  = frame.copy()
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    if state["roi"]:
        cv.polylines(vis, [np.array(state["roi"])], state["roi_ok"], (0, 0, 200), 2)
        for p in state["roi"]:
            draw_circle(vis, p, 5, (0, 0, 255))
    if len(state["split"]) == 2:
        cv.line(vis, state["split"][0], state["split"][1], (0, 220, 255), 2)
    if state["line_x"] is not None:
        cv.line(vis, (state["line_x"], 0), (state["line_x"], h), (0, 255, 255), 2)

    if state["roi_ok"]:
        roi = np.array(state["roi"], np.int32)
        bx, by, bw, bh = cv.boundingRect(roi)
        crop  = frame[by:by+bh, bx:bx+bw]
        rmask = np.zeros((bh, bw), np.uint8)
        cv.fillPoly(rmask, [roi - [bx, by]], 255)

        lr = 0 if paused else -1
        m  = bgsub.apply(crop, learningRate=lr)
        m  = cv.bitwise_and(m, m, mask=rmask)
        m  = (m == 255).astype(np.uint8) * 255
        m  = cv.morphologyEx(m, cv.MORPH_OPEN,
                cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
        close_k = max(1, int(panel["Close px"]))
        m  = cv.morphologyEx(m, cv.MORPH_CLOSE,
                cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_k, close_k)))
        m  = separate(m, panel["Fusion px"])
        mask[by:by+bh, bx:bx+bw] = m

    objs = detect(mask, panel["Area minima"])
    update_tracks(objs)

    for tid, t in tracks.items():
        cx, cy = t["cx"], t["cy"]
        bx, by, bw, bh = t["bx"], t["by"], t["bw"], t["bh"]
        cv.rectangle(vis, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
        draw_circle(vis, (cx, cy), 4, (255, 255, 0))
        cv.putText(vis, str(tid), (cx+6, cy-6),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    return vis, mask, objs

# ─────────────────────── COUNT ───────────────────────────

prev_pos = {}

def count_crossings():
    if not (state["roi_ok"] and state["split_ok"] and state["line_ok"]):
        prev_pos.clear()
        prev_pos.update({tid: t["cx"] for tid, t in tracks.items()})
        return

    xline = state["line_x"]
    for tid, t in tracks.items():
        cx, cy = t["cx"], t["cy"]
        if tid not in prev_pos:
            continue
        px = prev_pos[tid]
        if (px < xline <= cx) or (px > xline >= cx):
            side = "izq" if cy < split_y(cx) else "der"
            state["count"][side] += 1
            state["history"].append((time.time(), side))

    prev_pos.clear()
    prev_pos.update({tid: t["cx"] for tid, t in tracks.items()})

# ─────────────────────── CHART ───────────────────────────

def get_rates():
    now = time.time()
    t0  = state["start_time"]
    n   = int((now - t0) / BIN_S) + 1
    izq = [0] * n
    der = [0] * n
    for ts, side in state["history"]:
        idx = int((ts - t0) / BIN_S)
        if 0 <= idx < n:
            (izq if side == "izq" else der)[idx] += 1
    start_bin = max(0, n - 20)
    izq, der  = izq[-20:], der[-20:]
    factor    = 60 / BIN_S
    ri        = [c * factor for c in izq]
    rd        = [c * factor for c in der]
    labels    = [time.strftime("%H:%M", time.localtime(t0 + (start_bin + i) * BIN_S))
                 for i in range(len(ri))]
    return labels, ri, rd

def launch_chart():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig.patch.set_facecolor("#111")

    def style(ax, title):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#888", labelsize=8)
        ax.set_ylabel("veh/min", color="#888", fontsize=8)
        ax.set_title(title, color="white", fontsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.grid(color="#222", linewidth=0.5)

    def update(_):
        labels, ri, rd = get_rates()
        for ax, rates, color, title in (
            (ax1, ri, "#378ADD", f"← izquierda  (total: {state['count']['izq']})"),
            (ax2, rd, "#639922", f"derecha →  (total: {state['count']['der']})"),
        ):
            ax.cla()
            style(ax, title)
            ax.plot(labels, rates, color=color, linewidth=2)
            ax.fill_between(range(len(rates)), rates, alpha=0.15, color=color)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", color="#888", fontsize=8)
        fig.tight_layout()

    _ani = FuncAnimation(fig, update, interval=2000, cache_frame_data=False)
    plt.show()

# ─────────────────────── MAIN ────────────────────────────

def main():
    bgsub  = cv.createBackgroundSubtractorMOG2(500, 16, False)
    paused = False
    frozen = None

    threading.Thread(target=launch_chart, daemon=True).start()

    for key, frame in autoStream():
        if key == ord(" "):
            paused = not paused
            if paused:
                frozen = frame.copy()
        elif key == ord("r"):
            state["roi"], state["roi_ok"], state["mode"] = [], False, "roi"
        elif key == ord("d"):
            state["split"], state["split_ok"], state["mode"] = [], False, "split"
        elif key == ord("c"):
            state["line_x"], state["line_ok"], state["mode"] = None, False, "count"

        src = frozen if paused else frame
        vis, mask, objs = process(src, bgsub, paused)
        count_crossings()

        info = (
            f"AreaMin={panel['Area minima']:.0f} "
            f"Fusion={panel['Fusion px']:.0f} "
            f"Close={panel['Close px']:.0f} | "
            f"R=ROI D=Split C=Line | "
            f"<-:{state['count']['izq']} ->:{state['count']['der']}"
        )
        cv.imshow(WIN, draw_hud(vis, info))
        cv.imshow("mask", mask)
        panel.show()

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()