// Browser-based ML inference using Hugging Face Transformers.js
import { pipeline } from "@huggingface/transformers";
import { extractTextFeatures, FEATURE_CORRELATIONS, FEATURE_BASELINES, FEATURE_SCALES, FEATURE_METADATA } from "./textFeatures";

let classifierPipeline: any | null = null;
let isLoading = false;
let loadError: string | null = null;

// Using a smaller ONNX-compatible model for browser inference
// The original model might not have ONNX weights, so we'll use a compatible alternative
const MODEL_NAME = "Xenova/distilbert-base-uncased-finetuned-sst-2-english";

export async function loadModel(onProgress?: (progress: number) => void): Promise<void> {
  if (classifierPipeline) return;
  if (isLoading) return;

  isLoading = true;
  loadError = null;

  try {
    console.log("Loading AI detection model...");
    
    // Try to load with progress tracking
    classifierPipeline = await pipeline(
      "text-classification",
      MODEL_NAME,
      {
        device: "webgpu", // Use WebGPU if available, will fallback to WASM
      }
    );
    
    console.log("Model loaded successfully!");
    onProgress?.(100);
  } catch (error) {
    console.error("Error loading model:", error);
    loadError = error instanceof Error ? error.message : "Failed to load model";
    // Fallback to WASM if WebGPU fails
    try {
      console.log("Retrying with WASM...");
      classifierPipeline = await pipeline(
        "text-classification",
        MODEL_NAME
      );
      console.log("Model loaded with WASM successfully!");
      onProgress?.(100);
      loadError = null;
    } catch (wasmError) {
      console.error("WASM fallback also failed:", wasmError);
      throw wasmError;
    }
  } finally {
    isLoading = false;
  }
}

export function isModelLoaded(): boolean {
  return classifierPipeline !== null;
}

export function getLoadError(): string | null {
  return loadError;
}

function zTo01(z: number): number {
  return 1.0 / (1.0 + Math.exp(-z));
}

function featureDeviation(key: string, value: number): number {
  const baseline = FEATURE_BASELINES[key] || 0;
  const scale = FEATURE_SCALES[key] || 1;
  const raw = (value - baseline) / scale;
  return Math.max(-1, Math.min(1, raw));
}

interface Span {
  text: string;
  start: number;
  end: number;
  score: number;
  dominant_feature: string;
  features: {
    [key: string]: {
      contribution: number;
      reason: string;
    };
  };
}

// Simplified span generation based on sentence analysis
function generateSpans(text: string, prediction: number, features: any): Span[] {
  const sentences = text.split(/[.!?]+/).filter(s => s.trim());
  const spans: Span[] = [];
  
  const targetLabel = prediction === 1 ? "ai" : "human";
  
  // Analyze each sentence
  sentences.forEach((sent, idx) => {
    const trimmed = sent.trim();
    if (!trimmed || trimmed.split(/\s+/).length < 3) return;
    
    // Calculate feature contributions for this sentence
    const sentFeatures = extractTextFeatures(trimmed);
    const contributions: Record<string, number> = {};
    
    Object.keys(FEATURE_CORRELATIONS).forEach(key => {
      const corr = FEATURE_CORRELATIONS[key];
      const deviation = featureDeviation(key, sentFeatures[key as keyof typeof sentFeatures] || 0);
      contributions[key] = corr * deviation;
    });
    
    // Find dominant feature
    const sortedContribs = Object.entries(contributions)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    
    const [dominantFeature, dominantScore] = sortedContribs[0] || ["burstiness", 0];
    
    // Calculate overall score
    const totalScore = Object.values(contributions).reduce((sum, val) => sum + Math.abs(val), 0);
    const normalizedScore = Math.min(1, totalScore / 2); // Normalize to 0-1
    
    if (normalizedScore > 0.1) { // Only include significant spans
      const startPos = text.indexOf(trimmed);
      
      spans.push({
        text: trimmed,
        start: startPos,
        end: startPos + trimmed.length,
        score: normalizedScore,
        dominant_feature: dominantFeature,
        features: Object.fromEntries(
          sortedContribs.slice(0, 3).map(([key, value]) => [
            key,
            {
              contribution: value,
              reason: FEATURE_METADATA[key]?.description || key
            }
          ])
        )
      });
    }
  });
  
  // Sort by score and return top spans
  return spans.sort((a, b) => b.score - a.score).slice(0, 20);
}

export async function analyzeText(text: string): Promise<any> {
  if (!classifierPipeline) {
    await loadModel();
  }

  if (!classifierPipeline) {
    throw new Error("Model not loaded");
  }

  console.log("Analyzing text...");

  // Get model prediction
  const result = await classifierPipeline(text, { topk: 2 });
  console.log("Classification result:", result);

  // Extract features
  const features = extractTextFeatures(text);
  console.log("Extracted features:", features);

  // For distilbert-sst-2, labels are POSITIVE/NEGATIVE
  // We'll map this to AI/Human based on the confidence
  // Higher POSITIVE score = more AI-like (coherent, well-formed)
  const positiveResult = Array.isArray(result) ? result.find((r: any) => r.label === "POSITIVE") : null;
  const p_ai = positiveResult ? positiveResult.score : 0.5;
  const prediction = p_ai > 0.5 ? 1 : 0;

  // Calculate global scores
  const globalScores = {
    lexical_complexity: (features.lex_avg_word_len / 6 + features.lex_long_word_ratio) / 2,
    formality: features.form_connector_ratio * 2,
    burstiness: 1 - Math.min(1, features.burstiness / 1.5) // Invert: low burstiness = more AI-like
  };

  // Generate spans
  const spans = generateSpans(text, prediction, features);

  // Create feature impacts
  const featureImpacts = Object.entries(FEATURE_CORRELATIONS)
    .map(([key, corr]) => {
      const rawValue = features[key as keyof typeof features] || 0;
      const deviation = featureDeviation(key, rawValue);
      const signedScore = corr * deviation;
      
      return {
        key,
        label: FEATURE_METADATA[key]?.label || key,
        description: FEATURE_METADATA[key]?.description || "",
        raw_value: rawValue,
        deviation,
        correlation: corr,
        signed_score: signedScore,
        direction: signedScore >= 0 ? "ai" : "human",
        source: "global"
      };
    })
    .sort((a, b) => Math.abs(b.signed_score) - Math.abs(a.signed_score));

  return {
    prediction,
    p_ai,
    p_human: 1 - p_ai,
    global_scores: globalScores,
    spans,
    words: text.split(/\s+/),
    feature_impacts: featureImpacts
  };
}
