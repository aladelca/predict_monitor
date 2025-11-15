"""Utilidades de inferencia sobre el modelo multimodal."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
import sys
from types import ModuleType
from typing import Optional
from urllib.parse import urlparse

from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from torchvision import transforms
import requests
from requests import Session
from io import BytesIO

from .config import (
    DEFAULT_MODEL_PATH,
    DEVICE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SENTENCE_MODEL_NAME,
)
from .models import MultimodalRegressor

try:  # Torch < 2.6 no expone add_safe_globals
    from torch.serialization import add_safe_globals as _add_safe_globals
except ImportError:  # pragma: no cover - compatibilidad retro
    def _add_safe_globals(_):
        return None


_add_safe_globals([MultimodalRegressor])


def _register_pickle_aliases() -> None:
    """Asegura que el pickled `MultimodalRegressor` pueda resolverse."""

    for alias in ("__main__", "__mp_main__"):
        module = sys.modules.get(alias)
        if module is None:
            module = ModuleType(alias)
            sys.modules[alias] = module
        if not hasattr(module, "MultimodalRegressor"):
            setattr(module, "MultimodalRegressor", MultimodalRegressor)


_register_pickle_aliases()


class PricePredictor:
    """Carga el modelo y expone un método de predicción reutilizable."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        text_model_name: str = SENTENCE_MODEL_NAME,
        device: torch.device = DEVICE,
        auto_load: bool = True,
        http_timeout: int = 30,
        http_headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self.text_model_name = text_model_name
        self.device = device
        self.text_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[torch.nn.Module] = None
        self.text_encoder: Optional[SentenceTransformer] = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.http_timeout = http_timeout
        self.http: Session = requests.Session()
        if http_headers:
            self.http.headers.update(http_headers)
        self._lock = threading.Lock()
        if auto_load:
            self.load()

    def load(self) -> None:
        """Carga pesos y dependencias si aún no están inicializados."""
        print(self.model_path)
        with self._lock:
            if self.text_encoder is None:
                self.text_encoder = SentenceTransformer(self.text_model_name, device=self.text_device)
                self.text_encoder.eval()

            if self.model is None:
                embedding_dim = self.text_encoder.get_sentence_embedding_dimension()
                self.model = MultimodalRegressor(text_dim=embedding_dim)
                checkpoint = self._load_checkpoint()
                self._load_state_dict(checkpoint)
                self.model.to(self.device)
                self.model.eval()

    def _load_checkpoint(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de pesos en {self.model_path}")

        errors = []
        
        try:
            return self._call_torch_load(weights_only=False)
        except Exception as exc:  # pragma: no cover - logging propaga información
            logging.debug("Fallo al cargar checkpoint con weights_only=%s: %s", False, exc)
            errors.append(exc)
        raise RuntimeError(f"No se pudo cargar el checkpoint: {errors[-1] if errors else 'desconocido'}")

    def _call_torch_load(self, weights_only):
        kwargs = {"map_location": self.device}
        if weights_only is not None:
            kwargs["weights_only"] = weights_only
        try:
            return torch.load(self.model_path, **kwargs)
        except TypeError:
            kwargs.pop("weights_only", None)
            return torch.load(self.model_path, **kwargs)

    def _load_state_dict(self, checkpoint) -> None:
        if isinstance(checkpoint, MultimodalRegressor):
            state_dict = checkpoint.state_dict()
        elif isinstance(checkpoint, dict):
            state_dict = (
                checkpoint.get("state_dict")
                or checkpoint.get("model_state_dict")
                or checkpoint
            )
        else:
            raise TypeError("Formato de checkpoint no soportado")

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:  # pragma: no cover - aviso en caso de divergencias
            logging.warning("Missing keys: %s | Unexpected keys: %s", missing, unexpected)

    def predict(self, image_path: str | Path, description: str) -> float:
        if not description or not description.strip():
            raise ValueError("La descripción no puede estar vacía")

        self.load()

        if self.model is None or self.text_encoder is None:
            raise RuntimeError("El modelo no se cargó correctamente")

        clean_desc = " ".join(description.lower().split())
        text_tensor = self.text_encoder.encode(  # type: ignore[arg-type]
            [clean_desc],
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).to(self.device)

        image = self._load_image(image_path)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            prediction = self.model(image_tensor, text_tensor)
        return float(prediction.squeeze().item())

    def _is_remote_source(self, source: str | Path) -> bool:
        if not isinstance(source, str):
            return False
        scheme = urlparse(source).scheme.lower()
        return scheme in {"http", "https"}

    def _load_image(self, source: str | Path) -> Image.Image:
        if self._is_remote_source(source):
            response = self.http.get(str(source), timeout=self.http_timeout)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")

        img_path = Path(source)
        if not img_path.exists():
            raise FileNotFoundError(f"No existe la imagen en {img_path}")
        return Image.open(img_path).convert("RGB")
