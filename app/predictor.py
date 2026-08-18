"""Thread-safe, long-lived PyTorch prediction service."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import Settings
from app.preprocess import preprocess_image

logger = logging.getLogger(__name__)


class ModelNotReadyError(RuntimeError):
    """Raised when inference is requested before a model has loaded."""


class SignLanguagePredictor:
    """Loads a model once and exposes synchronized inference."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Any | None = None
        self.labels: list[str] | None = None
        self.device: Any | None = None
        self._lock = threading.Lock()

    def load_model(self) -> None:
        """Load a PyTorch model and optional human-readable class labels."""
        import torch

        if self.model is not None:
            return
        model_path = self._resolve_model_path()
        if self.settings.labels_path is not None:
            self.labels = self._load_labels(self.settings.labels_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = torch.jit.load(str(model_path), map_location=self.device)
            logger.info("Loaded TorchScript model from %s", model_path)
        except RuntimeError:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            state_dict = self._extract_state_dict(checkpoint)
            model = self._build_torchvision_model(self._class_count_from_state_dict(state_dict))
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"Checkpoint does not match {self.settings.model_architecture}. Missing: {missing}; unexpected: {unexpected}")
            logger.info("Loaded state-dict model from %s", model_path)
        self.model = model.to(self.device).eval()
        logger.info("Model loaded on %s; labels available: %s", self.device, self.labels is not None)

    def predict(self, image: Image.Image) -> tuple[str, int, bool, float]:
        """Run one image through the loaded model and return its verified class index."""
        if self.model is None or self.device is None:
            raise ModelNotReadyError("Model is not loaded.")
        import torch

        tensor = preprocess_image(image, self.settings.image_size).to(self.device)
        with self._lock, torch.inference_mode():
            output = self.model(tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            probabilities = torch.softmax(output, dim=1)
            confidence, index = torch.max(probabilities, dim=1)
        class_index = int(index.item())
        if self.labels is not None and class_index >= len(self.labels):
            raise RuntimeError("Model returned a class index not present in labels.json.")
        label_available = self.labels is not None
        prediction = self.labels[class_index] if label_available else f"class_{class_index}"
        return prediction, class_index, label_available, float(confidence.item())

    def _resolve_model_path(self) -> Path:
        model_path = self.settings.model_path
        if not model_path.exists():
            legacy = Path(__file__).resolve().parent.parent / "efficientnet_asl_weights.pth"
            if legacy.exists():
                model_path = legacy
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.settings.model_path}")
        return model_path

    @staticmethod
    def _load_labels(path: Path) -> list[str]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        labels = raw if isinstance(raw, list) else raw.get("labels")
        if not isinstance(labels, list) or not labels or not all(isinstance(label, str) and label for label in labels):
            raise ValueError("labels.json must be a non-empty JSON array of strings, or an object with a labels array.")
        return labels

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a state dictionary or contain model_state_dict/state_dict.")
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
        if not all(isinstance(key, str) for key in state_dict):
            raise ValueError("Invalid checkpoint state dictionary.")
        return {key.removeprefix("module."): value for key, value in state_dict.items()}

    @staticmethod
    def _class_count_from_state_dict(state_dict: dict[str, Any]) -> int:
        """Infer output count from the standard torchvision EfficientNet classifier."""
        classifier_weight = state_dict.get("classifier.1.weight")
        if classifier_weight is None or getattr(classifier_weight, "ndim", 0) != 2:
            raise ValueError("Cannot infer class count from checkpoint; provide a compatible torchvision EfficientNet state dictionary.")
        return int(classifier_weight.shape[0])

    def _build_torchvision_model(self, num_classes: int):
        from torchvision import models

        try:
            model = models.get_model(self.settings.model_architecture, weights=None, num_classes=num_classes)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Unsupported MODEL_ARCHITECTURE: {self.settings.model_architecture}") from exc
        return model
