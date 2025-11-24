# Backend API for AI Text Detector

This backend provides AI text detection with explainability using a RoBERTa-based model.

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the FastAPI server:

**Default (using uvicorn):**
```bash
python main.py
```
This will automatically use uvicorn with auto-reload enabled.

**Or using uvicorn directly:**
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 5000 --reload
```

**Environment variables:**
- `PORT` - Server port (default: 5000)
- `DEBUG` - Enable debug mode with detailed logging (default: false)
- `RELOAD` - Enable auto-reload on code changes (default: true, or follows DEBUG setting)

The API will start on `http://localhost:5000`

You can also access the interactive API documentation at `http://localhost:5000/docs`

## Troubleshooting

### Backend not responding

1. **Check if the backend is running:**
   ```bash
   # Test the health endpoint
   curl http://localhost:5000/health
   # Should return: {"status":"healthy"}
   ```

2. **Check if the port is available:**
   ```bash
   # On Windows
   netstat -ano | findstr :5000
   # On Linux/Mac
   lsof -i :5000
   ```

3. **Verify model loading:**
   - Check the console output when starting the server
   - You should see "Loading AI detector model..." followed by "Model loaded successfully!"
   - If you see errors, make sure all dependencies are installed

4. **Test the API directly:**
   ```bash
   # Test the analyze endpoint
   curl -X POST http://localhost:5000/api/analyze \
     -H "Content-Type: application/json" \
     -d '{"text": "This is a test sentence."}'
   ```

### Frontend connection issues

1. **Set the API URL:**
   - Create a `.env` file in the project root with:
     ```
     VITE_API_URL=http://localhost:5000
     ```
   - Or the frontend will default to `http://localhost:5000`

2. **Check CORS:**
   - The backend is configured to allow all origins (`*`)
   - If you still have CORS issues, check the browser console

3. **Verify both servers are running:**
   - Backend: `python backend/main.py` (should be on port 5000)
   - Frontend: `npm run dev` (should be on port 8080)

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
