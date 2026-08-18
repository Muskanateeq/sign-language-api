---
title: Sign Language Recognition API
emoji: 🤟
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# Sign Language Recognition API

A FastAPI service that accepts a PNG/JPEG image and returns a sign-language prediction from a PyTorch model. The model is loaded once at startup and inference is synchronized for safe shared use.

## Setup

Use Python 3.12 or newer from this directory.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put the model artifact in `model/`:

- `model/efficientnet_asl_weights.pth`

The supplied checkpoint is a 29-output torchvision EfficientNet state dictionary. It does **not** include class names, so the API now starts and returns the verified index as `class_0` through `class_28`. To return sign names, add the original training label map at `model/labels.json` and set `LABELS_PATH=model/labels.json` in `.env`. Its exact output-label order is essential. Set `MODEL_ARCHITECTURE` and preprocessing values in `.env` if your training setup differs from the default torchvision `efficientnet_b0` at 224×224 with ImageNet normalization.

## Run

```powershell
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for Swagger UI or `/redoc` for ReDoc.

## Docker

The Docker image includes the required `model/efficientnet_asl_weights.pth` checkpoint and uses CPU PyTorch. Build and start the production container from this directory:

```powershell
docker compose up --build -d
```

The API is available at `http://localhost:8000`; verify its container health with:

```powershell
docker compose ps
curl.exe http://localhost:8000/health
```

To stop it:

```powershell
docker compose down
```

For a direct Docker command instead of Compose:

```powershell
docker build -t sign-language-api:latest .
docker run --rm -p 8000:8000 --env-file .env --name sign-language-api sign-language-api:latest
```

## CI/CD: GitHub Actions to Hugging Face Spaces

The workflow at `.github/workflows/ci-cd.yml` runs on each pull request and push to `main`. It validates Python syntax, loads the EfficientNet checkpoint, and builds the Docker image. A successful push to `main` then syncs this repository to a Hugging Face **Docker Space**.

Before your first push, create a Hugging Face access token with write access to the target Space. In the GitHub repository’s **Settings → Secrets and variables → Actions**, add:

| Type | Name | Value |
| --- | --- | --- |
| Secret | `HF_TOKEN` | Hugging Face write/fine-grained token scoped to the Space |
| Variable | `HF_SPACE_ID` | Hugging Face Space ID, for example `your-hf-username/sign-language-api` |

The workflow creates/syncs the target Space as a Docker Space. The front matter at the top of this README tells Hugging Face to route the Space to this FastAPI service on port `8000`.

The model file is 16 MB, so Git LFS is required before you commit it:

```powershell
git lfs install
git add .gitattributes model/efficientnet_asl_weights.pth
git add .
git commit -m "Add CI/CD deployment to Hugging Face Space"
git push origin main
```

After the deployment job succeeds, your service is available at `https://<your-hf-username>-<space-name>.hf.space/docs`.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Returns `{"status":"running"}` |
| GET | `/health` | Returns `{"status":"healthy"}` |
| POST | `/predict` | Predicts a sign from a multipart `image` field |

Example request:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -F "image=@C:\path\to\image.jpg"
```

Example response:

```json
{
  "success": true,
  "prediction": "class_4",
  "class_index": 4,
  "label_available": false,
  "confidence": 0.98,
  "processing_time_ms": 34.2
}
```

Only `.png`, `.jpg`, and `.jpeg` images are accepted. Uploads are capped at 10 MiB and 20 million pixels by default. Errors use `{ "success": false, "message": "..." }` and appropriate HTTP status codes.

## Layout

```text
app/       API routes, validation, configuration, and inference service
model/     model checkpoint and label map
uploads/   reserved for optional future persisted uploads; requests are not saved
```

## Deployment notes

Set a restrictive `ALLOWED_ORIGINS` value for production, for example `["https://example.com"]`. Model availability is logged during startup; `/predict` returns 503 rather than serving a partially initialized model if its artifacts are absent or incompatible.
