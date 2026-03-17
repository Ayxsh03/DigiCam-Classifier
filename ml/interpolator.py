"""
Latent Space Interpolation Module
Uses a pre-trained PyTorch VAE to interpolate between digits in latent space.
"""

import os
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import imageio.v2 as imageio

import config


# ─── VAE Architecture ───────────────────────────────────────────────────────────

class MNISTVAE(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 28 * 28)

    def encode(self, x):
        x = x.view(-1, 28 * 28)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        out = torch.sigmoid(self.fc5(h))
        return out.view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decode(z), mu, logvar

    def encode_latent(self, x):
        mu, _ = self.encode(x)
        return mu

    def decode_latent(self, z):
        return self.decode(z)


# ─── Interpolator ────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LatentInterpolator:
    def __init__(self):
        self.model = MNISTVAE(latent_dim=10).to(DEVICE)
        self.model.load_state_dict(
            torch.load(config.VAE_MODEL_PATH, map_location=DEVICE)
        )
        self.model.eval()
        self.transform = transforms.ToTensor()
        print("[Interpolator] PyTorch VAE model loaded successfully.")

    def _load_sample(self, digit: int) -> torch.Tensor:
        """Load a sample MNIST image for a given digit label."""
        path = os.path.join(config.MNIST_SAMPLES_DIR, f"imag_{digit}.png")
        img = Image.open(path).convert("L").resize((28, 28))
        return self.transform(img).unsqueeze(0).to(DEVICE)

    def _load_image_bytes(self, image_bytes: bytes) -> torch.Tensor:
        """Load an image from raw bytes."""
        img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
        return self.transform(img).unsqueeze(0).to(DEVICE)

    def interpolate_labels(self, digit_a: int, digit_b: int, steps: int = 20) -> list:
        """Interpolate between two digit labels using sample images."""
        img_a = self._load_sample(digit_a)
        img_b = self._load_sample(digit_b)
        return self._interpolate(img_a, img_b, steps)

    def interpolate_images(self, img_a_bytes: bytes, img_b_bytes: bytes, steps: int = 20) -> list:
        """Interpolate between two uploaded images."""
        img_a = self._load_image_bytes(img_a_bytes)
        img_b = self._load_image_bytes(img_b_bytes)
        return self._interpolate(img_a, img_b, steps)

    def _interpolate(self, img_a: torch.Tensor, img_b: torch.Tensor, steps: int) -> list:
        """Core interpolation logic."""
        with torch.no_grad():
            z1 = self.model.encode_latent(img_a)
            z2 = self.model.encode_latent(img_b)

        frames = []
        with torch.no_grad():
            for i in range(steps):
                alpha = i / (steps - 1)
                z = (1 - alpha) * z1 + alpha * z2
                recon = self.model.decode_latent(z)
                img = recon.squeeze().cpu().numpy()
                if img.max() <= 1.0:
                    img = img * 255.0
                frames.append(img.astype(np.uint8))

        return frames

    def create_gif(self, frames: list, duration: float = 0.08) -> bytes:
        """Convert list of numpy frames into GIF bytes."""
        buf = io.BytesIO()
        # Forward + reverse for smooth loop
        looped = frames + frames[::-1][1:-1]
        imageio.mimsave(buf, looped, format="GIF", duration=duration, loop=0)
        buf.seek(0)
        return buf.read()
