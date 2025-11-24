"""
Flask API for AI Text Detector with Explainability
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from explainable import (
    load_detector,
    explain_text_with_features,
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model on startup
print("Loading AI detector model...")
try:
    load_detector(use_gpu=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200


@app.route("/api/analyze", methods=["POST"])
def analyze_text():
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
        data = request.get_json()
        
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data["text"]
        if not text or not text.strip():
            return jsonify({"error": "Empty text provided"}), 400
        
        max_length = data.get("max_length", 512)
        
        # Run the explainable analysis with top_k=20 for comprehensive highlighting
        result = explain_text_with_features(
            text=text,
            max_length=max_length,
            top_k=20
        )
        
        # Convert words to token format for frontend compatibility
        tokens = []
        for word in result.get("words", []):
            # Simple score assignment based on spans
            score = 0.5  # default neutral
            features = []
            
            # Check if word is in any span
            for span in result.get("spans", []):
                if word in span["text"]:
                    score = min(span["score"] / 10.0, 1.0)  # normalize
                    features.append(span["dom_feature"])
            
            tokens.append({
                "text": word,
                "score": score,
                "features": features if features else ["neutral"]
            })
        
        # Format response for frontend
        response = {
            "prediction": result["prediction"],
            "p_ai": result["p_ai"],
            "confidence": result["p_ai"],  # frontend expects this
            "global_scores": result["global_scores"],
            "spans": result["spans"],
            "words": result["words"],
            "tokens": tokens
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error analyzing text: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
