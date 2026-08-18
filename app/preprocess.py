"""Image validation and model input preprocessing."""
from __future__ import annotations

from io import BytesIO

import numpy as np
from fastapi import HTTPException, status
from PIL import Image, UnidentifiedImageError

from app.config import Settings

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_ALLOWED_FORMATS = {"JPEG", "PNG"}
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def validate_filename(filename: str | None) -> None:
    """Reject missing names and unsupported file extensions."""
    if not filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Image file is required.")
    if filename.lower().rsplit(".", 1)[-1] not in {"jpg", "jpeg", "png"}:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only PNG, JPG and JPEG files are allowed.")


def decode_image(data: bytes, settings: Settings) -> Image.Image:
    """Decode and validate untrusted image data without writing it to disk."""
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Image file is required.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Image file is too large.")
    try:
        with Image.open(BytesIO(data)) as probe:
            probe.verify()
        image = Image.open(BytesIO(data))
        if image.format not in _ALLOWED_FORMATS:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only PNG, JPG and JPEG files are allowed.")
        if image.width * image.height > settings.max_image_pixels:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Image dimensions are too large.")
        return image.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid or corrupted image file.") from exc


def preprocess_image(image: Image.Image, image_size: int):
    """Convert a PIL image to normalized NCHW PyTorch input."""
    import torch

    resized = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0)
