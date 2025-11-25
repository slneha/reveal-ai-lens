"""
Flask API for AI Text Detector with Explainability
Supports both WSGI (Flask built-in) and ASGI (uvicorn) via asgiref wrapper
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
# Configure CORS explicitly to handle preflight OPTIONS requests
# Allow all origins, headers, and methods for development
CORS(app, 
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     supports_credentials=False,
     automatic_options=True)

# Create ASGI wrapper for uvicorn compatibility
try:
    from asgiref.wsgi import WsgiToAsgi
    asgi_app = WsgiToAsgi(app)
except ImportError:
    # asgiref not available, ASGI wrapper won't work
    asgi_app = None

# Add after_request handler to ensure CORS headers are always set
# This is critical for OPTIONS preflight requests and works with both WSGI and ASGI
@app.after_request
def after_request(response):
    # Add CORS headers to all responses (including errors)
    # This ensures OPTIONS preflight requests always get proper headers
    origin = request.headers.get('Origin')
    if origin:
        # Use the requesting origin if provided
        response.headers.add('Access-Control-Allow-Origin', origin)
    else:
        # Fallback to * for requests without Origin header
        response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET, PUT, POST, DELETE, OPTIONS')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

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


@app.route("/api/analyze", methods=["POST", "OPTIONS"])
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
    # Handle OPTIONS preflight request - flask-cors should handle this automatically
    # but we'll handle it explicitly to be safe
    if request.method == "OPTIONS":
        return jsonify({}), 200
    
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

        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error analyzing text: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run Flask app with built-in WSGI server
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask server on port {port}...")
    print("Note: If you want to use uvicorn, run: uvicorn backend.main:asgi_app --host 0.0.0.0 --port 5000")
    app.run(host="0.0.0.0", port=port, debug=True)
