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

# Load the model on startup with memory optimizations
print("Loading AI detector model with memory optimizations (float16, low_cpu_mem_usage)...")
print("Note: Target is <512MB, but RoBERTa models typically require ~800-900MB even with optimizations")
try:
    load_detector(use_gpu=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200


@app.route("/api/memory", methods=["GET"])
def memory_info():
    """Get current memory usage information"""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        # Get system memory info
        sys_mem = psutil.virtual_memory()
        
        return jsonify({
            "process_memory_mb": mem_info.rss / (1024 * 1024),
            "process_memory_percent": process.memory_percent(),
            "system_total_mb": sys_mem.total / (1024 * 1024),
            "system_available_mb": sys_mem.available / (1024 * 1024),
            "system_used_percent": sys_mem.percent,
            "within_512mb_limit": (mem_info.rss / (1024 * 1024)) < 512
        }), 200
    except ImportError:
        return jsonify({
            "error": "psutil not available",
            "install": "pip install psutil"
        }), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
        top_k = data.get("top_k", 20)
        
        # Run the explainable analysis
        result = explain_text_with_features(
            text=text,
            max_length=max_length,
            top_k=top_k
        )
        
        # Memory cleanup after analysis
        import gc
        gc.collect()

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
    # Run Flask app
    # Railway/Render use PORT env var (auto-set), fallback to 5000 for local dev
    port = int(os.environ.get("PORT", 5000))
    # Disable debug mode in production (only enable for local dev on port 5000)
    debug = (port == 5000 and os.environ.get("FLASK_ENV") != "production")
    app.run(host="0.0.0.0", port=port, debug=debug)
