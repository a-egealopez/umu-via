#Exporta: SliderPanel, draw_line, draw_circle, draw_text, draw_hud

from __future__ import annotations
import cv2 as cv
import numpy as np
from config import UI_THEME

T = UI_THEME  # alias local para no repetir UI_THEME["..."] en todo el fichero


# ── SliderPanel ───────────────────────────────────────────────────────────────
class SliderPanel:

    def __init__(self, win_name: str, sliders: list[tuple]) -> None:
        self.SLIDER_H = T["SLIDER_H"]
        self.MARGIN   = T["MARGIN"]
        self.W        = T["WIDTH"]
        self.TRACK_W  = self.W - 2 * self.MARGIN

        self.win = win_name
        self.data: list[dict] = [
            {"label": label, "vmin": float(vmin), "vmax": float(vmax),
             "val": float(init), "unit": unit, "y": 15 + i * self.SLIDER_H}
            for i, (label, vmin, vmax, init, unit) in enumerate(sliders)
        ]
        self.H = len(self.data) * self.SLIDER_H + 20

        self._cache:     np.ndarray | None = None
        self._last_vals: tuple | None      = None
        self._dragging:  int | None        = None

        cv.namedWindow(self.win)
        cv.setMouseCallback(self.win, self._mouse)

    def __getitem__(self, label: str) -> float:
        return self._find(label)["val"]

    def show(self) -> None:
        vals = tuple(s["val"] for s in self.data)
        if vals != self._last_vals:
            self._cache     = self._draw()
            self._last_vals = vals
        cv.imshow(self.win, self._cache)

    def _find(self, label: str) -> dict:
        for s in self.data:
            if s["label"] == label:
                return s
        raise KeyError(f"Slider '{label}' no encontrado")

    def _draw(self) -> np.ndarray:
        img = np.full((self.H, self.W, 3), T["C_BG"], dtype=np.uint8)

        for s in self.data:
            ty   = s["y"] + 18
            frac = (s["val"] - s["vmin"]) / max(s["vmax"] - s["vmin"], 1e-6)
            fx   = self.MARGIN + int(frac * self.TRACK_W)

            # Etiqueta y valor
            cv.putText(img, s["label"], (self.MARGIN, s["y"] + 12),
                       cv.FONT_HERSHEY_DUPLEX, 0.42, T["C_LABEL"], 1, cv.LINE_AA)
            val_str = f"{s['val']:.0f}{s['unit']}"
            (tw, _), _ = cv.getTextSize(val_str, cv.FONT_HERSHEY_DUPLEX, 0.42, 1)
            cv.putText(img, val_str, (self.W - self.MARGIN - tw, s["y"] + 12),
                       cv.FONT_HERSHEY_DUPLEX, 0.42, T["C_VALUE"], 1, cv.LINE_AA)

            # Carril
            cv.rectangle(img, (self.MARGIN, ty - 2), (self.MARGIN + self.TRACK_W, ty + 2), T["C_TRACK"], -1)
            cv.rectangle(img, (self.MARGIN, ty - 2), (fx, ty + 2), T["C_FILL"], -1)

            # Marcas min/max
            cv.putText(img, f"{s['vmin']:.0f}", (self.MARGIN, ty + 14),
                       cv.FONT_HERSHEY_DUPLEX, 0.32, T["C_TICK"], 1, cv.LINE_AA)
            max_str = f"{s['vmax']:.0f}"
            (mw, _), _ = cv.getTextSize(max_str, cv.FONT_HERSHEY_DUPLEX, 0.32, 1)
            cv.putText(img, max_str, (self.MARGIN + self.TRACK_W - mw, ty + 14),
                       cv.FONT_HERSHEY_DUPLEX, 0.32, T["C_TICK"], 1, cv.LINE_AA)

            # Thumb
            cv.circle(img, (fx, ty), 7, T["C_THUMB"], -1, cv.LINE_AA)
            cv.circle(img, (fx, ty), 7, T["C_FILL"],   1, cv.LINE_AA)

        return img

    def _mouse(self, event: int, x: int, y: int, flags: int, _: object) -> None:
        if event == cv.EVENT_LBUTTONDOWN:
            for i, s in enumerate(self.data):
                if self.MARGIN <= x <= self.MARGIN + self.TRACK_W and abs(y - (s["y"] + 18)) < 12:
                    self._dragging = i
                    break
        elif event == cv.EVENT_MOUSEMOVE and self._dragging is not None:
            s = self.data[self._dragging]
            s["val"] = s["vmin"] + (s["vmax"] - s["vmin"]) * float(np.clip((x - self.MARGIN) / self.TRACK_W, 0, 1))
        elif event == cv.EVENT_LBUTTONUP:
            self._dragging = None


# ── Helpers de dibujo ─────────────────────────────────────────────────────────
def draw_line(frame, a, b, color=None, thickness=1) -> None:
    cv.line(frame, tuple(np.int32(a)), tuple(np.int32(b)),
            color or T["C_LINE"], thickness, cv.LINE_AA)


def draw_circle(frame, center, radius=5, color=None, thickness=-1) -> None:
    cv.circle(frame, tuple(np.int32(center)), radius,
              color or T["C_LINE"], thickness, cv.LINE_AA)


def draw_text(frame, text, pos, color=None, scale=0.5, thickness=1) -> None:
    cv.putText(frame, str(text), tuple(np.int32(pos)),
               cv.FONT_HERSHEY_DUPLEX, scale, color or T["C_TEXT"], thickness, cv.LINE_AA)


def draw_hud(frame, text, color=None) -> np.ndarray:
    _, w = frame.shape[:2]
    lines = str(text).split("  ")
    mid   = len(lines) // 2
    line1 = "  ".join(lines[:mid])
    line2 = "  ".join(lines[mid:])
    hud = np.zeros((52, w, 3), np.uint8)
    cv.putText(hud, line1, (8, 18),
               cv.FONT_HERSHEY_DUPLEX, 0.5, color or T["C_TEXT"], 1, cv.LINE_AA)
    cv.putText(hud, line2, (8, 42),
               cv.FONT_HERSHEY_DUPLEX, 0.5, color or T["C_TEXT"], 1, cv.LINE_AA)
    return np.vstack([frame, hud])