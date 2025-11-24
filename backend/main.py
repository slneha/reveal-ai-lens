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

        words = result.get("words", [])
        word_importance = result.get("word_importance", [0.0] * len(words))
        word_total_strength = result.get("word_total_strength", [0.0] * len(words))
        word_total_opposing = result.get("word_total_opposing", [0.0] * len(words))
        word_support_ai = result.get("word_support_ai", [0.0] * len(words))
        word_support_human = result.get("word_support_human", [0.0] * len(words))
        word_contrib_signed = result.get("word_contrib_signed", [0.0] * len(words))
        lex_contrib = result.get("lex_contrib", [0.0] * len(words))
        form_contrib = result.get("form_contrib", [0.0] * len(words))
        burst_contrib = result.get("burst_contrib", [0.0] * len(words))

        max_word_strength = max(word_total_strength) if word_total_strength else 0.0
        denom_strength = max_word_strength or 1.0

        tokens = []
        for i, word in enumerate(words):
            signed_score = float(word_contrib_signed[i]) if i < len(word_contrib_signed) else 0.0
            raw_strength = float(word_total_strength[i]) if i < len(word_total_strength) else 0.0
            normalized_strength = raw_strength / denom_strength if denom_strength > 0 else 0.0
            word_len = max(len(word.strip()), 1)
            length_weight = 0.35 + 0.65 * min(1.0, word_len / 12.0)
            score = float(min(1.0, normalized_strength * length_weight))

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
                    "signed_score": signed_score,
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
            "spans_opposing": result.get("spans_opposing", []),
            "words": result["words"],
            "tokens": tokens,
            "sentences": result.get("sentences", []),
            "sentences_opposing": result.get("sentences_opposing", []),
            "feature_impacts": result.get("feature_impacts", []),
            "word_contrib_signed": word_contrib_signed,
            "word_total_strength": word_total_strength,
            "word_total_opposing": word_total_opposing,
            "word_support_ai": word_support_ai,
            "word_support_human": word_support_human,
            "word_saliency_ai": result.get("word_saliency_ai", []),
            "word_saliency_human": result.get("word_saliency_human", []),
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
