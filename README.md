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

1. **Install Python dependencies:**
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Install Node dependencies:**
   ```bash
   npm install
   ```

3. **Start both frontend and backend:**
   ```bash
   npm run dev
   ```
   This will automatically start:
   - Frontend (Vite) on `http://localhost:8080`
   - Backend (Uvicorn) on `http://localhost:5000` with auto-reload enabled

**Alternative: Run separately**

- Frontend only: `npm run dev:frontend`
- Backend only: `npm run dev:backend` (uses uvicorn with auto-reload)

