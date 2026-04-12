#!/usr/bin/env python

from __future__ import annotations
import urllib.request
from pathlib import Path

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

# ── Configuración ─────────────────────────────────────────────────────────────
_MODEL_FILE = Path(__file__).parent / "embedder.tflite"
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite"
)


# ── Factoría ──────────────────────────────────────────────────────────────────
def create() -> EmbeddingMethod:
    if not _MODEL_FILE.exists():
        print(f"[INFO] Descargando modelo en {_MODEL_FILE} …")
        _MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_FILE))
        print("[INFO] Descarga completada.")

    options = vision.ImageEmbedderOptions(
        base_options=python.BaseOptions(model_asset_path=str(_MODEL_FILE)),
        l2_normalize=True,
        quantize=True,
    )
    return EmbeddingMethod(vision.ImageEmbedder.create_from_options(options))


# ── Clase principal ───────────────────────────────────────────────────────────
class EmbeddingMethod:
    name = "embedding"

    def __init__(self, embedder: vision.ImageEmbedder) -> None:
        self._embedder = embedder

    def _embed(self, img_bgr):
        rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._embedder.embed(mpimg).embeddings[0]

    def precompute(self, img_bgr):
        return self._embed(img_bgr)

    def compare(self, frame_bgr, descriptor) -> float:
        return float(vision.ImageEmbedder.cosine_similarity(self._embed(frame_bgr), descriptor))

    def best_match(self, frame_bgr, descriptors) -> tuple[int, list[float]]:
        sims = [self.compare(frame_bgr, d) for d in descriptors]
        return int(max(enumerate(sims), key=lambda x: x[1])[0]), sims