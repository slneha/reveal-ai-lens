# Backend API for AI Text Detector

This backend provides AI text detection with explainability using a RoBERTa-based model.

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask server:
```bash
python main.py
```

The API will start on `http://localhost:5000`

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
