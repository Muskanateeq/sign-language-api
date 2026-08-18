"""HTTP route handlers."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.preprocess import decode_image, validate_filename
from app.predictor import ModelNotReadyError, SignLanguagePredictor
from app.schemas import ErrorResponse, PredictionResponse, StatusResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=StatusResponse)
def root() -> StatusResponse:
    return StatusResponse(status="running")


@router.get("/health", response_model=StatusResponse)
def health() -> StatusResponse:
    return StatusResponse(status="healthy")


@router.post("/predict", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def predict(request: Request, image: UploadFile = File(...)) -> PredictionResponse:
    """Validate an upload and return its most likely sign class."""
    started = time.perf_counter()
    validate_filename(image.filename)
    raw = await image.read()
    await image.close()
    decoded = decode_image(raw, request.app.state.settings)
    predictor: SignLanguagePredictor = request.app.state.predictor
    try:
        label, class_index, label_available, confidence = predictor.predict(decoded)
    except ModelNotReadyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model is not ready. Check server logs.") from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Prediction failed.") from exc
    elapsed = round((time.perf_counter() - started) * 1000, 2)
    logger.info("Prediction completed: label=%s confidence=%.4f time_ms=%.2f", label, confidence, elapsed)
    return PredictionResponse(prediction=label, class_index=class_index, label_available=label_available, confidence=confidence, processing_time_ms=elapsed)
