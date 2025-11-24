# Explainable AI Text Detector (Browser Edition) 🚀

An advanced AI text detection application with comprehensive explainability features, now running **entirely in your browser** using WebGPU/WASM!

## ✨ What's New

This version has been **migrated from Python backend to browser-based ML inference**:
- ✅ **No backend required** - runs 100% in the browser
- ✅ **WebGPU accelerated** - fast inference with hardware acceleration
- ✅ **Complete privacy** - all processing happens locally, no data sent anywhere
- ✅ **Works offline** - once the model loads, no internet needed
- ✅ **Zero hosting costs** - deploy as a static site
- ✅ **Instant deployment** - publish with one click

## 🎯 Features

### Real-time AI Detection
- Binary classification: Human vs AI-generated text
- Confidence scores with visual gauge
- Instant analysis after model loads

### Explainable AI
- **Token-level highlighting** - See which phrases contribute to classification
- **Feature analysis** - Lexical diversity, formality, burstiness metrics
- **Dominant feature detection** - Understand the "why" behind each classification
- **Visual overlays** - Color-coded spans showing AI-supporting regions

### Linguistic Feature Analysis
The app analyzes multiple linguistic dimensions:
- **Lexical Complexity**: Long/technical vocabulary usage
- **Formality Level**: Formal connectors vs casual language
- **Burstiness**: Sentence length variance (lower = more AI-like)
- **Syntactic Patterns**: Comma density, sentence structure
- **First-Person Usage**: Personal pronoun frequency

## 🛠 Technology Stack

- **Frontend**: React + TypeScript + Vite
- **UI**: Tailwind CSS + shadcn/ui components  
- **ML**: Hugging Face Transformers.js
- **Inference**: WebGPU with WASM fallback
- **Acceleration**: Browser-native GPU computing

## 🚀 Getting Started

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

The app will automatically download the ML model on first visit (may take 30-60 seconds). Once loaded, all analysis runs instantly in your browser!

## 📊 How It Works

### 1. Model Loading
Downloads a lightweight ONNX-optimized transformer model to your browser cache

### 2. Feature Extraction  
Analyzes the text for:
- Lexical features (word length, vocabulary diversity)
- Syntactic features (sentence structure, punctuation)
- Formality markers (connectors, pronouns, contractions)
- Burstiness (sentence length variation)

### 3. Classification
Uses the transformer model to predict AI vs Human likelihood

### 4. Explainability Generation
- Identifies text spans that contribute to the classification
- Calculates feature contributions
- Highlights dominant features for each span
- Provides natural language explanations

## 🎨 Research Background

This implementation is based on empirical analysis showing strong correlations between:
- **Lexical complexity** (0.37 correlation with AI)
- **Burstiness** (-0.38 correlation - lower variance = more AI-like)
- **Formality markers** and assertiveness patterns
- **Stopword distribution** and topic uniformity

These linguistic features contextualize the model's gradient-based predictions.

## 📦 Deployment

### Lovable (One-Click)
Simply click **Publish** in the Lovable editor - no configuration needed!

### Other Platforms
Deploy to any static hosting service:
- Netlify
- Vercel  
- GitHub Pages
- Cloudflare Pages

All you need is the built static files - no backend setup required!

## 🔒 Privacy

**Your data never leaves your device.** All ML inference happens locally in the browser. No text is sent to any server, ensuring complete privacy.

## 🧪 Legacy Python Backend

The original Python backend using PyTorch and RoBERTa is preserved in the `/backend` folder for reference. However, it's no longer needed since the app now runs entirely in the browser.

To use the Python backend (not recommended):
```bash
pip install -r backend/requirements.txt
python backend/main.py
```

## 📝 License

This project uses open-source ML models and libraries. Check individual component licenses for details.

---

**Built with ❤️ using Lovable and Hugging Face Transformers.js**
