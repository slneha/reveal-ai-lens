/**
 * Browser-based ML inference using Hugging Face Inference API
 * Uses backend proxy to avoid CORS issues
 */

const BACKEND_API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

/**
 * Call Hugging Face Inference API via backend proxy (avoids CORS)
 */
async function callHFInferenceAPI(text: string, maxRetries = 3): Promise<any> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch(`${BACKEND_API_URL}/api/hf-inference`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      if (response.status === 503) {
        // Model is loading, wait and retry
        const waitTime = Math.pow(2, attempt) * 5; // Exponential backoff: 5s, 10s, 20s
        console.log(`Model loading, waiting ${waitTime}s... (attempt ${attempt + 1}/${maxRetries})`);
        await new Promise(resolve => setTimeout(resolve, waitTime * 1000));
        continue;
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(`HF API error: ${response.status} ${errorData.error || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      if (attempt === maxRetries - 1) {
        throw error;
      }
      const waitTime = Math.pow(2, attempt) * 2;
      console.log(`Retrying in ${waitTime}s... (attempt ${attempt + 1}/${maxRetries})`);
      await new Promise(resolve => setTimeout(resolve, waitTime * 1000));
    }
  }
  
  throw new Error("Failed to call HF Inference API after retries");
}

/**
 * Run inference and get predictions using HF Inference API
 */
export async function predict(text: string): Promise<{
  prediction: number; // 0 = Human, 1 = AI
  p_human: number;
  p_ai: number;
  logits: number[];
}> {
  console.log("Calling Hugging Face Inference API...");
  
  const result = await callHFInferenceAPI(text);

  // Process results - HF returns array of {label, score} or single object
  let p_human = 0.5;
  let p_ai = 0.5;

  const results = Array.isArray(result) ? result : [result];

  for (const item of results) {
    const label = (item.label || "").toLowerCase();
    const score = item.score || 0.5;
    
    if (label.includes("human") || label === "label_0" || label === "0" || label.includes("negative")) {
      p_human = score;
    } else if (label.includes("ai") || label.includes("machine") || label === "label_1" || label === "1" || label.includes("positive")) {
      p_ai = score;
    }
  }

  // If we only got one result, infer the other
  if (results.length === 1 && p_human === 0.5 && p_ai === 0.5) {
    const singleScore = results[0].score || 0.5;
    const singleLabel = (results[0].label || "").toLowerCase();
    if (singleLabel.includes("human") || singleLabel === "label_0" || singleLabel === "0") {
      p_human = singleScore;
      p_ai = 1 - singleScore;
    } else {
      p_ai = singleScore;
      p_human = 1 - singleScore;
    }
  }

  // Normalize to sum to 1
  const total = p_human + p_ai;
  if (total > 0) {
    p_human = p_human / total;
    p_ai = p_ai / total;
  }

  const prediction = p_ai > p_human ? 1 : 0;
  const logits = [Math.log(p_human / (1 - p_human)), Math.log(p_ai / (1 - p_ai))];

  return {
    prediction,
    p_human,
    p_ai,
    logits,
  };
}

/**
 * Get token-level saliency scores using heuristics
 * Note: Since we're using HF Inference API, we can't compute gradients.
 * We use linguistic heuristics as a proxy for saliency, which the backend
 * will combine with its own feature analysis for better accuracy.
 */
export async function getTokenSaliency(
  text: string
): Promise<{
  tokens: string[];
  scores: number[];
}> {
  // Use heuristic-based saliency (backend will refine this with its feature analysis)
  const words = text.split(/\s+/);
  const scores: number[] = [];
  
  // Heuristic: longer words, formal connectors, and words in middle tend to be more important
  const midPoint = words.length / 2;
  const formalConnectors = new Set([
    "furthermore", "moreover", "therefore", "consequently", "thus", "hence",
    "additionally", "nevertheless", "nonetheless", "accordingly", "subsequently"
  ]);
  
  for (let i = 0; i < words.length; i++) {
    const word = words[i].toLowerCase().replace(/[.,!?;:]/g, "");
    let score = 0.3; // Base score
    
    // Longer words get higher score (lexical complexity indicator)
    if (word.length >= 7) score += 0.2;
    if (word.length >= 10) score += 0.15;
    
    // Formal connectors get higher score (formality indicator)
    if (formalConnectors.has(word)) score += 0.3;
    
    // Words near middle get slightly higher score (often more important contextually)
    const distanceFromMid = Math.abs(i - midPoint) / Math.max(midPoint, 1);
    score += (1 - distanceFromMid) * 0.1;
    
    scores.push(Math.min(1.0, score));
  }
  
  // Normalize scores
  const maxScore = Math.max(...scores, 0.01);
  for (let i = 0; i < scores.length; i++) {
    scores[i] = scores[i] / maxScore;
  }
  
  return {
    tokens: words,
    scores,
  };
}

/**
 * Combined function: get both prediction and saliency
 */
export async function analyzeWithSaliency(text: string): Promise<{
  prediction: number;
  p_human: number;
  p_ai: number;
  tokens: string[];
  tokenScores: number[];
}> {
  const [predictionResult, saliencyResult] = await Promise.all([
    predict(text),
    getTokenSaliency(text),
  ]);

  return {
    ...predictionResult,
    tokens: saliencyResult.tokens,
    tokenScores: saliencyResult.scores,
  };
}
