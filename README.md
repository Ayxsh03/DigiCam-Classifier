# DigiCam — Classify & Morph Digits

<p align="center">
  <img src="static/assets/lsi_demo.gif" alt="Latent Space Interpolation Demo" width="280"/>
</p>
<p align="center"><em>Smooth morphing between digits via Latent Space Interpolation</em></p>

A deployable web application featuring two deep learning modules for handwritten digits:

| Module | Model | What It Does |
|---|---|---|
| **🔍 Classification** | Keras CNN | Draw or upload a digit → get a real-time prediction with confidence scores |
| **🌀 Interpolation** | PyTorch VAE | Pick two digits → see a smooth morphing animation through latent space |

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/Ayxsh03/DigiCam-Classifier.git
cd DigiCam-Classifier

# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 📂 Project Structure

```text
DigiCam-Classifier/
├── app.py                  # Flask server (API routes)
├── config.py               # Configuration & model paths
├── ml/
│   ├── classifier.py       # Keras CNN — digit prediction
│   └── interpolator.py     # PyTorch VAE — latent interpolation + GIF
├── models/
│   ├── my_model.keras      # Pre-trained CNN weights
│   └── mnist_vae.pth       # Pre-trained VAE weights
├── templates/
│   └── index.html          # Single-page frontend
├── static/
│   ├── css/style.css       # Dark theme + glassmorphism
│   ├── js/app.js           # Canvas drawing, API calls
│   └── assets/             # Demo GIF, sample MNIST images
├── Procfile                # Gunicorn start command
├── render.yaml             # Render deployment blueprint
└── requirements.txt
```

---

## 🧠 Architecture

### Classification Module
- **Model:** Sequential CNN trained on MNIST (28×28 grayscale)
- **Pipeline:** Image → Grayscale → Resize 28×28 → Normalize → CNN → Softmax probabilities
- **Endpoint:** `POST /api/classify`

### Interpolation Module
- **Model:** Variational Autoencoder (784 → 512 → 256 → **10D latent** → 256 → 512 → 784)
- **Pipeline:** Two images → Encode to latent μ → Linear interpolation → Decode each step → GIF
- **Loss:** BCE + KL Divergence
- **Endpoint:** `POST /api/interpolate`

---

## ☁️ Deployment

### Render (Recommended)
1. Push to GitHub
2. Connect repo on [render.com](https://render.com)
3. It auto-detects `render.yaml` — deploy with one click

### Manual
```bash
pip install gunicorn
gunicorn app:app --bind 0.0.0.0:8000 --timeout 120
```

---

## 🛠️ Tech Stack

**Backend:** Python · Flask · Gunicorn
**ML:** TensorFlow/Keras · PyTorch
**Frontend:** HTML5 Canvas · Vanilla CSS · Vanilla JS
**Deploy:** Render

---

## 📝 License

MIT License

## 🙏 Acknowledgments

- **MNIST Dataset** — Yann LeCun et al.
- **PyTorch** & **TensorFlow/Keras** — Deep learning frameworks
