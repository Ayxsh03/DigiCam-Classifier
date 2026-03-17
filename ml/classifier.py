"""
Digit Classification Module
Uses a pre-trained Keras CNN to classify handwritten digits (0-9).
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import io
import config


def _build_model():
    """Rebuild the same architecture that was used to train the model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class DigitClassifier:
    def __init__(self):
        # Try direct load first; fall back to rebuilding architecture + loading weights
        try:
            self.model = tf.keras.models.load_model(config.KERAS_MODEL_PATH)
        except Exception:
            print("[Classifier] Direct load failed, rebuilding architecture and loading weights...")
            self.model = _build_model()
            donor = tf.keras.models.load_model(
                config.KERAS_MODEL_PATH, compile=False,
                custom_objects={"softmax_v2": tf.keras.activations.softmax},
            )
            self.model.set_weights(donor.get_weights())
            del donor
        print("[Classifier] Keras CNN model loaded successfully.")

    def predict(self, image_bytes: bytes) -> dict:
        """
        Accepts raw image bytes, preprocesses, and returns prediction.
        Returns: {"digit": int, "confidence": float, "probabilities": list}
        """
        # Open image and preprocess
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = img.resize((28, 28))
        img_array = np.array(img, dtype=np.float32)

        # Invert if background is white (white = 255)
        if np.mean(img_array) > 127:
            img_array = 255.0 - img_array

        # Normalize to [0, 1]
        img_array = img_array / 255.0

        # Reshape for model: (1, 28, 28)
        img_array = img_array.reshape(1, 28, 28)

        # Predict
        predictions = self.model.predict(img_array, verbose=0)
        probabilities = predictions[0].tolist()
        digit = int(np.argmax(probabilities))
        confidence = float(probabilities[digit])

        return {
            "digit": digit,
            "confidence": round(confidence, 4),
            "probabilities": [round(p, 4) for p in probabilities],
        }

