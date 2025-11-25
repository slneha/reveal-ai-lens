# Explainable AI-vs-Human Text Detector  
RoBERTa + Gradient-Based Saliency + Linguistic Feature Attribution

This project implements an interpretable classifier that distinguishes between human-written and machine-generated text using the model  
`andreas122001/roberta-mixed-detector` and a custom explanation module.

The objective is not only to output a class label, but to provide a clear justification of the decision by identifying the specific phrases that influenced the model toward the AI or human class.

---

## Research Motivation

Before implementation, an empirical analysis was conducted on a dataset of AI-generated and human-generated text. The goal was to identify latent linguistic variables with the strongest statistical correlation to the model’s probability output (`p_ai`).

The following factors exhibited the highest correlation:

- Lexical complexity (long or technical vocabulary)
- Formality markers
- Burstiness / variance in sentence length
- Assertiveness vs. hedging language
- Topic uniformity across sentences
- Stopword distribution patterns

These correlated variables were incorporated into the interpretability layer to contextualize the gradient-based saliency results. They do not drive the final classification but help users understand why the model favored the AI-generated or human-generated label.

---

## Key Features

| Component | Purpose |
|----------|----------|
| RoBERTa classifier | Binary detection: 0 = Human, 1 = AI |
| Gradient-based saliency (primary signal) | Highlights the exact phrases that increase the AI or human logit |
| Correlation-based linguistic insights (secondary signal) | Adds descriptive interpretation without interfering with gradients |
| Span-level explanation | Identifies influential multi-word segments rather than isolated tokens |
| Notebook and API support | Works with Pandas DataFrames or raw text |

---

## Why Gradients 
The explanation framework uses:

‖ ∂(logit_AI) / ∂(embedding(token)) ‖

This measures how much each token influences the AI logit. Gradient saliency therefore provides a more direct and faithful attribution than attention or handcrafted features alone. Linguistic variables are included separately to support interpretability but do not override gradient signals.

---

## To Run

### Backend

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Run Flask server (default: http://localhost:5000)
python backend/main.py
```

### Frontend

```bash
# Install dependencies
npm install

# Configure API URL (optional, defaults to http://localhost:5000)
# Create .env file with:
# VITE_API_URL=http://localhost:5000
# Or for production:
# VITE_API_URL=https://web-production-26e24.up.railway.app

# Run development server
npm run dev
```

### Production Backend

The backend is deployed on Railway at: `https://web-production-26e24.up.railway.app`

To use the production backend in the frontend, set:
```bash
VITE_API_URL=https://web-production-26e24.up.railway.app
```

