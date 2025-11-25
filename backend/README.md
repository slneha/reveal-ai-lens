# Backend API for AI Text Detector

This backend provides AI text detection with explainability using a RoBERTa-based model.

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask server:

**Option 1: Flask built-in server (Recommended):**
```bash
python main.py
```

**Option 2: With uvicorn (requires asgiref):**
```bash
uvicorn backend.main:asgi_app --host 0.0.0.0 --port 5000 --reload
```

The API will start on `http://localhost:5000`

**Note:** Flask is a WSGI application. If you see errors about "missing start_response", make sure you're either:
- Using `python main.py` (Flask's built-in WSGI server)
- Using `uvicorn backend.main:asgi_app` (with ASGI wrapper)
- NOT using `uvicorn backend.main:app` (this won't work with Flask)

## API Endpoints

### POST /api/analyze
Analyzes text for AI detection with explainability features.

**Request Body:**
```json
{
  "text": "Your text here...",
  "top_k": 3,
  "max_length": 512
}
```

**Response:**
```json
{
  "prediction": 1,
  "p_ai": 0.85,
  "confidence": 0.85,
  "global_scores": {
    "lexical_complexity": 0.72,
    "formality": 0.68,
    "burstiness": 0.45
  },
  "spans": [...],
  "words": [...],
  "tokens": [...]
}
```

### GET /health
Health check endpoint that returns the API status.

## Model

Uses `andreas122001/roberta-mixed-detector` from Hugging Face for AI text detection.
