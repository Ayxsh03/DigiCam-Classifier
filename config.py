import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
KERAS_MODEL_PATH = os.path.join(BASE_DIR, "models", "my_model.keras")
VAE_MODEL_PATH = os.path.join(BASE_DIR, "models", "mnist_vae.pth")

# MNIST sample images for interpolation
MNIST_SAMPLES_DIR = os.path.join(BASE_DIR, "static", "assets", "MNIST")

# Upload config
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB

# Interpolation defaults
DEFAULT_STEPS = 20
MAX_STEPS = 40
