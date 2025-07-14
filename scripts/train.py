# scripts/train.py
"""
Hinglish quick‑use:
python scripts/train.py --config config/config.yaml
"""

from pathlib import Path
import sys, argparse, yaml, os

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import tensorflow as tf
from src.data.dataset_loader import load_dataset
from src.models.mask_detector import build_model
from src.utils.logger import logger
from src.utils.exceptions import wrap_error


@wrap_error
def train(cfg_path: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    p, h = cfg["paths"], cfg["model"]

    ds_train, ds_val = load_dataset(
        p["dataset_root"], h["img_size"], h["batch_size"]
    )

    model = build_model(h["img_size"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(h["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    os.makedirs(p["model_dir"], exist_ok=True)
    ckpt = os.path.join(p["model_dir"], "mask_cnn_best.h5")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt, monitor="val_accuracy", save_best_only=True, mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
    ]

    logger.info("🚀 Training shuru…")
    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=h["epochs"],
        callbacks=callbacks,
    )
    model.save(ckpt)
    logger.success("🎉 Model saved -> %s", ckpt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    train(args.config)
