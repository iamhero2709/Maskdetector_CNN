# serving/app.py
from __future__ import annotations
from pathlib import Path
import sys, os, io

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

# ─── Project imports — make root importable first ───────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent  # …/Mask-Detection
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.utils.logger import logger
from src.utils.exceptions import wrap_error

# ─── Robust model path (local dev vs Docker) ────────────────────────────────────
MODEL_PATH = Path("saved_models/mask_cnn_best.h5")  # Local dev
if not MODEL_PATH.exists():                         # Docker path
    MODEL_PATH = Path("/models/mask_cnn_best.h5")
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"❌ Cannot find trained model at {MODEL_PATH}. "
        "Run scripts/train.py first or mount the model into /models."
    )

# ─── Load model once at startup ─────────────────────────────────────────────────
logger.info("🔄 Loading model from %s …", MODEL_PATH)
MODEL = tf.keras.models.load_model(str(MODEL_PATH))
IMG_SIZE = 128  # Match your training size

# ─── FastAPI setup ─────────────────────────────────────────────────────────────
app = FastAPI(title="Mask Detection API", version="1.0.0")


class Prediction(BaseModel):
    mask: bool
    probability: float


@wrap_error
@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    """Return mask / no‑mask prediction for a JPEG/PNG image."""
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    img_bytes = await file.read()
    img = (
        Image.open(io.BytesIO(img_bytes))
        .convert("RGB")
        .resize((IMG_SIZE, IMG_SIZE))  # ✅ Proper tuple
    )
    arr = np.expand_dims(np.array(img) / 255.0, 0)  # Normalize
    prob = float(MODEL(arr, training=False).numpy()[0][0])
    logger.info("🔍 Prediction served (prob=%.3f)", prob)
    return {"mask": prob >= 0.5, "probability": prob}
