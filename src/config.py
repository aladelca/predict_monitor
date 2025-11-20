"""Configuración centralizada para la inferencia del modelo."""

from __future__ import annotations

import os
from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_model_path() -> Path:
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return Path(env_path)

    candidates = [
        BASE_DIR / "artifacts" / "model.pth",
        BASE_DIR / "model.pth",
        BASE_DIR / "model_weights.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_MODEL_PATH = _resolve_model_path()


def get_device() -> torch.device:
    """Selecciona el mejor dispositivo disponible."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_available = getattr(torch.backends, "mps", None)
    if mps_available and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
SENTENCE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
IMAGE_SIZE = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
