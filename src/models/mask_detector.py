# src/models/mask_detector.py
"""
Hinglish summary:
Yeh simple CNN hai:
Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → Flatten → Dense → Softmax (2 classes)
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from src.utils.logger import logger
from src.utils.exceptions import wrap_error


@wrap_error
def build_model(img_size: int = 128) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(img_size, img_size, 3)),

            # Conv Block 1
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(),

            # Conv Block 2
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(),

            # Conv Block 3
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),

            # 2‑class softmax (mask / nomask)
            layers.Dense(2, activation="softmax"),
        ]
    )

    logger.info("✅ CNN model ready (Conv‑ReLU‑Pool‑Flatten‑Softmax)")
    return model
