# CPU-only production image for the FastAPI sign-language inference service.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install dependencies before application code so this layer is reusable.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# A non-root account limits the impact of a compromised API process.
RUN addgroup --system appuser && adduser --system --ingroup appuser appuser

COPY app ./app
COPY model ./model
COPY uploads ./uploads
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Uses only the standard library; no extra health-check package is required.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

# One worker keeps one copy of the PyTorch model in memory.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
