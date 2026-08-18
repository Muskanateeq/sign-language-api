"""FastAPI application entry point."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.predictor import SignLanguagePredictor
from app.routes import router


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.settings = settings
    app.state.predictor = SignLanguagePredictor(settings)
    try:
        app.state.predictor.load_model()
    except Exception:
        logging.getLogger(__name__).exception("Model could not be loaded; prediction endpoint will return 503")
    logging.getLogger(__name__).info("API started (%s)", settings.environment)
    yield


settings = get_settings()
app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False if settings.allowed_origins == ["*"] else True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"success": False, "message": str(exc.detail)})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    missing_image = any(error.get("loc", [])[-1:] == ["image"] for error in exc.errors())
    code = 400 if missing_image else 422
    message = "Image file is required." if missing_image else "Invalid request."
    return JSONResponse(status_code=code, content={"success": False, "message": message})


app.include_router(router)
