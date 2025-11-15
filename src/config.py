"""Configuración centralizada para la inferencia del modelo."""

from pathlib import Path
import torch


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "model.pth"


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

