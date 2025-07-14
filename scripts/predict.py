#!/usr/bin/env python
"""
Predict mask / no‑mask for a single image.

Example
-------
python scripts/predict.py \
       --image sample.jpg \
       --model saved_models/maskdet_best.h5
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent  # …/Mask-Detection
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
import tensorflow as tf
from PIL import Image
from src.utils.logger import logger

# ────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────
def load_image(image_path: Path, img_size: int = 224) -> np.ndarray:
    """Load and preprocess a single image for inference."""
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    arr = np.array(img) / 255.0  # scale to [0,1]
    return np.expand_dims(arr, axis=0)  # add batch dimension


def predict(image_path: str, model_path: str, img_size: int = 224) -> None:
    image_path = Path(image_path)
    model_path = Path(model_path)

    if not image_path.exists():
        logger.error("❌ Image not found: %s", image_path)
        return
    if not model_path.exists():
        logger.error("❌ Model not found: %s", model_path)
        return

    # Load model once
    logger.info("🔄 Loading model from %s …", model_path)
    model = tf.keras.models.load_model(model_path)

    # Preprocess image
    logger.info("🖼️  Preprocessing image %s …", image_path.name)
    inp = load_image(image_path, img_size)

    # Predict
    logger.info("🔍 Running inference …")
    prob = float(model(inp, training=False).numpy()[0][0])
    has_mask = prob >= 0.5

    logger.success("✅ Result: %s (prob=%.3f)", "MASK" if has_mask else "NO MASK", prob)


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask‑Detection inference")
    parser.add_argument("--image", required=True, help="Path to .jpg/.png image")
    parser.add_argument(
        "--model",
        default="saved_models/maskdet_best.h5",  # adjust if different
        help="Path to trained .h5 model",
    )
    parser.add_argument("--img-size", type=int, default=224, help="Image square size")
    args = parser.parse_args()

    predict(args.image, args.model, args.img_size)
