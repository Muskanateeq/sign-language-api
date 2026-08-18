"""Pydantic response models."""
from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    status: str


class PredictionResponse(BaseModel):
    success: bool = True
    prediction: str
    class_index: int = Field(ge=0)
    label_available: bool
    confidence: float = Field(ge=0, le=1)
    processing_time_ms: float = Field(ge=0)


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
