# """
# DigiCam-Classifier Web Application
# Flask server serving two ML modules:
#   1. /api/classify  — CNN digit classification
#   2. /api/interpolate — VAE latent space interpolation
# """

# import os
# import base64
# from flask import Flask, render_template, request, jsonify, send_file
# import io

# import config
# from ml.classifier import DigitClassifier
# from ml.interpolator import LatentInterpolator

# app = Flask(__name__)
# app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# # ─── Load models at startup ─────────────────────────────────────────────────────
# print("Loading ML models...")
# classifier = DigitClassifier()
# interpolator = LatentInterpolator()
# print("All models loaded. Server ready.")


# # ─── Routes ──────────────────────────────────────────────────────────────────────

# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/api/classify", methods=["POST"])
# def classify():
#     """
#     Accepts either:
#       - A file upload (multipart form, field name "image")
#       - A JSON body with {"image": "<base64 data URL>"}
#     Returns JSON with prediction.
#     """
#     try:
#         image_bytes = None

#         if request.content_type and "multipart" in request.content_type:
#             file = request.files.get("image")
#             if file:
#                 image_bytes = file.read()
#         else:
#             data = request.get_json(silent=True)
#             if data and "image" in data:
#                 # Strip data URL prefix if present
#                 b64 = data["image"]
#                 if "," in b64:
#                     b64 = b64.split(",", 1)[1]
#                 image_bytes = base64.b64decode(b64)

#         if not image_bytes:
#             return jsonify({"error": "No image provided"}), 400

#         result = classifier.predict(image_bytes)
#         return jsonify(result)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route("/api/interpolate", methods=["POST"])
# def interpolate():
#     """
#     Accepts JSON: {"digit_a": int, "digit_b": int, "steps": int}
#     Returns a GIF as binary.
#     """
#     try:
#         data = request.get_json(silent=True)
#         if not data:
#             return jsonify({"error": "No JSON body provided"}), 400

#         digit_a = int(data.get("digit_a", 0))
#         digit_b = int(data.get("digit_b", 1))
#         steps = min(int(data.get("steps", config.DEFAULT_STEPS)), config.MAX_STEPS)

#         if not (0 <= digit_a <= 9 and 0 <= digit_b <= 9):
#             return jsonify({"error": "Digits must be between 0 and 9"}), 400

#         frames = interpolator.interpolate_labels(digit_a, digit_b, steps)
#         gif_bytes = interpolator.create_gif(frames)

#         return send_file(
#             io.BytesIO(gif_bytes),
#             mimetype="image/gif",
#             download_name=f"morph_{digit_a}_to_{digit_b}.gif",
#         )

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5001)

"""
DigiCam-Classifier Web Application
Flask server serving two ML modules:
  1. /api/classify  — CNN digit classification
  2. /api/interpolate — VAE latent space interpolation
"""

import os
import base64
import threading
import io
from flask import Flask, render_template, request, jsonify, send_file

import config
from ml.classifier import DigitClassifier
from ml.interpolator import LatentInterpolator

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# ─── Lazy-loaded models ─────────────────────────────────────────────────────────
classifier = None
interpolator = None


def load_models():
    global classifier, interpolator
    if classifier is None or interpolator is None:
        print("🚀 Loading ML models...")
        classifier = DigitClassifier()
        interpolator = LatentInterpolator()
        print("✅ Models loaded successfully.")


# Optional: preload in background (non-blocking)
def preload_models():
    load_models()


threading.Thread(target=preload_models).start()


# ─── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/classify", methods=["POST"])
def classify():
    try:
        load_models()

        image_bytes = None

        if request.content_type and "multipart" in request.content_type:
            file = request.files.get("image")
            if file:
                image_bytes = file.read()
        else:
            data = request.get_json(silent=True)
            if data and "image" in data:
                b64 = data["image"]
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                image_bytes = base64.b64decode(b64)

        if not image_bytes:
            return jsonify({"error": "No image provided"}), 400

        result = classifier.predict(image_bytes)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/interpolate", methods=["POST"])
def interpolate():
    try:
        load_models()

        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        digit_a = int(data.get("digit_a", 0))
        digit_b = int(data.get("digit_b", 1))
        steps = min(int(data.get("steps", config.DEFAULT_STEPS)), config.MAX_STEPS)

        if not (0 <= digit_a <= 9 and 0 <= digit_b <= 9):
            return jsonify({"error": "Digits must be between 0 and 9"}), 400

        frames = interpolator.interpolate_labels(digit_a, digit_b, steps)
        gif_bytes = interpolator.create_gif(frames)

        return send_file(
            io.BytesIO(gif_bytes),
            mimetype="image/gif",
            download_name=f"morph_{digit_a}_to_{digit_b}.gif",
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Health check (helps Render detect port quickly) ────────────────────────────
@app.route("/health")
def health():
    return {"status": "ok"}
