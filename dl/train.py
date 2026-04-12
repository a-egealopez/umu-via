#!/usr/bin/env python

from __future__ import annotations
from pathlib import Path

from ultralytics import YOLO

# ── Rutas ─────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent
DATA_YAML  = _HERE / "house_objects.yolov8" / "data.yaml"
BASE_MODEL = "yolo11n.pt"

# ── Hiperparámetros ───────────────────────────────────────────────────────────
EPOCHS = 100
IMGSZ  = 640

# Aumentación de color
HSV_H = 0.3
HSV_S = 0.7
HSV_V = 0.4

# Aumentación geométrica
FLIPLR    = 0.5
DEGREES   = 15
SCALE     = 0.5
TRANSLATE = 0.2


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"No se encontró el fichero de datos: {DATA_YAML}\n"
            "Asegúrate de tener el dataset en dl/house_objects.yolov8/"
        )

    model = YOLO(BASE_MODEL)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        augment=True,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        fliplr=FLIPLR,
        degrees=DEGREES,
        scale=SCALE,
        translate=TRANSLATE,
    )


if __name__ == "__main__":
    main()