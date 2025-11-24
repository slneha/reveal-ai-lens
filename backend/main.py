"""
FastAPI for AI Text Detector with Explainability
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import sys
import os
import uvicorn

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from explainable import (
    load_detector,
    explain_text_with_features,
)

app = FastAPI(
    title="AI Text Detector API",
    description="AI Text Detector with Explainability",
    version="1.0.0"
)

# Enable CORS for all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    top_k: Optional[int] = Field(3, description="Number of top spans to return")
    max_length: Optional[int] = Field(512, description="Maximum text length")


class HealthResponse(BaseModel):
    status: str


@app.on_event("startup")
def load_model():
    """Load the model on startup"""
    print("Loading AI detector model...")
    try:
        load_detector(use_gpu=False)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Text Detector API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/api/analyze")
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze text for AI detection with explainability
    
    Expected JSON body:
    {
        "text": "string",
        "top_k": int (optional, default 3),
        "max_length": int (optional, default 512)
    }
    
    Returns:
    {
        "prediction": int (0=Human, 1=AI),
        "p_ai": float (probability of AI),
        "confidence": float (same as p_ai, for frontend compatibility),
        "global_scores": {
            "lexical_complexity": float,
            "formality": float,
            "burstiness": float
        },
        "spans": [
            {
                "start": int,
                "end": int,
                "text": string,
                "score": float,
                "dom_feature": string,
                "reason": string
            }
        ],
        "words": [string],
        "tokens": [
            {
                "text": string,
                "score": float,
                "features": [string]
            }
        ]
    }
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")
        
        max_length = request.max_length or 512
        
        # Run the explainable analysis with top_k=20 for comprehensive highlighting
        result = explain_text_with_features(
            text=request.text,
            max_length=max_length,
            top_k=20
        )

        words = result.get("words", [])
        word_importance = result.get("word_importance", [0.0] * len(words))
        lex_contrib = result.get("lex_contrib", [0.0] * len(words))
        form_contrib = result.get("form_contrib", [0.0] * len(words))
        burst_contrib = result.get("burst_contrib", [0.0] * len(words))

        tokens = []
        for i, word in enumerate(words):
            score = float(word_importance[i]) if i < len(word_importance) else 0.0

            feats = []
            if i < len(lex_contrib) and lex_contrib[i] > 0:
                feats.append("lexical_complexity")
            if i < len(form_contrib) and form_contrib[i] > 0:
                feats.append("formality")
            if i < len(burst_contrib) and burst_contrib[i] > 0:
                feats.append("burstiness")
            if not feats:
                feats = ["neutral"]

            tokens.append(
                {
                    "text": word,
                    "score": score,
                    "features": feats,
                }
            )

        response = {
            "prediction": result["prediction"],
            "p_ai": result["p_ai"],
            "p_human": result.get("p_human"),
            "confidence": result["p_ai"],
            "global_scores": result["global_scores"],
            "spans": result["spans"],
            "words": result["words"],
            "tokens": tokens,
            "sentences": result.get("sentences", []),
            "feature_impacts": result.get("feature_impacts", []),
        }

        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error analyzing text: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Use uvicorn (works on Windows and all platforms)
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    reload = os.environ.get("RELOAD", "true" if debug else "false").lower() == "true"
    
    print(f"Starting server with uvicorn on port {port}...")
    if reload:
        print("Auto-reload enabled (restart on code changes)")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level="debug" if debug else "info"
    )
