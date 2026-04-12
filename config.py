from pathlib import Path

# ── Cámara ────────────────────────────────────────────────────────────────────
CAMERA_URL = "http://100.81.54.29:4747/video.mjpg"

# ── Calibración ───────────────────────────────────────────────────────────────
CALIB_FILE = Path(__file__).parent / "calibracion" / "calib.txt"
FRAME_W    = 640
FRAME_H    = 480

# ── Clasificador ──────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "classifier" / "models"

# ── Detección de actividad ────────────────────────────────────────────────────
EVENTS_DIR   = Path(__file__).parent / "actividad" / "events"
PRE_SEC      = 2    # segundos de pre-buffer antes del evento
POST_SEC     = 1    # segundos tras el último frame con movimiento
MIN_AREA_MOV = 800  # área mínima (px²) para considerar movimiento real

# ── UI / Tema ─────────────────────────────────────────────────────────────────
UI_THEME = {
    "SLIDER_H": 52,
    "MARGIN":   30,
    "WIDTH":    380,

    # Colores BGR
    "C_BG":     (28, 22, 18),
    "C_TRACK":  (72, 55, 45),
    "C_FILL":   (225, 153, 66),
    "C_THUMB":  (255, 255, 255),
    "C_LABEL":  (192, 174, 160),
    "C_VALUE":  (255, 204, 102),
    "C_TICK":   (104, 85, 74),
    "C_LINE":   (225, 153, 66),
    "C_TEXT":   (247, 242, 237),
    "C_HUD":    (44, 32, 26),
    "C_ACCENT": (100, 100, 255),

    # Tipografía
    "F_FONT":    2,     # cv2.FONT_HERSHEY_DUPLEX
    "F_SCALE_L": 0.45,  # escala etiquetas
    "F_SCALE_V": 0.55,  # escala valores
    "F_THICK":   1,
    "F_LTYPE":   16,    # cv2.LINE_AA
}