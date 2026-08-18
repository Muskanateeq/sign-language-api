# Prompt for Codex CLI

You are a Senior AI/ML Engineer and FastAPI Backend Architect.

Your task is to build a production-ready FastAPI REST API for a Sign Language Recognition model.

## Project Overview

I have already trained a Sign Language Recognition model using Python. The model predicts the sign shown in an input image.

The API should allow a client application (React, Next.js, Mobile App, etc.) to upload an image and receive the predicted sign as a JSON response.

The code should follow clean architecture, modular programming, proper error handling, and production-ready practices.

---

# Goal

Build a FastAPI backend that exposes REST endpoints for Sign Language prediction.

Flow:

Client
↓

Upload Image

↓

FastAPI API

↓

Preprocess Image

↓

Load Trained Model

↓

Predict Sign

↓

Return JSON Response

---

# Tech Stack

- Python 3.12+
- FastAPI
- Uvicorn
- Pydantic
- OpenCV
- Pillow (PIL)
- NumPy
- Torch (if model is PyTorch)
- TensorFlow/Keras (if model is TensorFlow)
- Python typing
- Logging
- dotenv (if needed)

Use whichever framework matches the trained model.

---

# Folder Structure

Generate the project with this structure.

```text
sign-language-api/
│
├── app/
│   ├── main.py
│   ├── routes.py
│   ├── predictor.py
│   ├── preprocess.py
│   ├── schemas.py
│   ├── config.py
│   ├── utils.py
│   └── __init__.py
│
├── model/
│   ├── sign_model.pt
│   ├── labels.json
│   └── README.md
│
├── uploads/
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

# API Endpoint

Create this endpoint:

POST /predict

Request

multipart/form-data

Field:

image

Example:

```
POST /predict
Content-Type: multipart/form-data

image = user_image.jpg
```

---

# Prediction Pipeline

The API should perform these steps:

1. Receive uploaded image

2. Validate image format

Accept:

- jpg
- jpeg
- png

Reject invalid files.

3. Read image safely

4. Preprocess image

The preprocessing function should be isolated.

Typical preprocessing may include:

- resize
- normalization
- RGB conversion
- tensor conversion

Make this easily configurable.

5. Load trained model only once

The model must be loaded during application startup.

Never reload model for every request.

6. Perform inference

Use:

```
torch.no_grad()
```

if PyTorch.

7. Get predicted class

8. Convert class index into sign label

using

labels.json

9. Return response.

---

# Response

Example

```json
{
  "success": true,
  "prediction": "HELLO",
  "confidence": 0.98,
  "processing_time_ms": 34
}
```

---

# Error Responses

Return proper HTTP status codes.

Example

Unsupported file

```json
{
    "success": false,
    "message": "Only PNG, JPG and JPEG files are allowed."
}
```

Missing image

```json
{
    "success": false,
    "message": "Image file is required."
}
```

Prediction failure

```json
{
    "success": false,
    "message": "Prediction failed."
}
```

---

# Health Endpoint

Create

GET /

Returns

```json
{
    "status":"running"
}
```

Also create

GET /health

Returns

```json
{
    "status":"healthy"
}
```

---

# Model Loading

Create a dedicated predictor class.

Example

```python
class SignLanguagePredictor:

    def __init__(self):
        ...

    def load_model(self):
        ...

    def predict(self, image):
        ...
```

Load model only once.

---

# Preprocessing

Create separate module

preprocess.py

Example

```python
def preprocess_image(image):
    ...
```

No preprocessing logic should exist inside the API route.

---

# Validation

Validate:

- Empty file
- Invalid image
- Corrupted image
- Wrong extension
- Huge image size

Return proper HTTP exceptions.

---

# Logging

Use Python logging.

Log:

- API started
- Model loaded
- Prediction completed
- Errors
- Processing time

---

# Performance

Optimize for inference.

Requirements:

- Load model once
- No duplicate preprocessing
- Efficient memory usage
- Thread-safe inference
- Fast response

---

# API Documentation

Use FastAPI automatic docs.

Swagger

```
/docs
```

Redoc

```
/redoc
```

---

# Code Quality

Generate production-quality code.

Requirements:

- Type hints
- Docstrings
- Modular design
- SOLID principles
- Clean naming
- Comments only where needed
- No duplicated code

---

# Requirements.txt

Generate a complete requirements.txt.

---

# README

Generate a professional README containing:

- Installation
- Virtual environment
- Dependency installation
- Running the server
- API endpoints
- Example requests
- Example responses
- Project structure

---

# Bonus Features

If possible, also implement:

✅ Confidence score

✅ Prediction time

✅ Request logging

✅ CORS middleware

✅ Environment variables

✅ Docker-ready structure

✅ Unit-test friendly architecture

---

# Important

Do not generate placeholder code.

Implement every file completely.

Use best practices suitable for production deployment.

The API must be ready to run using:

```bash
uvicorn app.main:app --reload
```

The final output should include all source code, folder structure, requirements.txt, README.md, and any configuration files necessary to run the API immediately after placing the trained model and labels into the `model/` directory.