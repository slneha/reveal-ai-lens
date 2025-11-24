import { CSSProperties, useState } from "react";
import { TokenTooltip } from "./TokenTooltip";

export interface SpanData {
  start: number;
  end: number;
  text: string;
  score: number;
  dom_feature: string;
  reason: string;
  lex_contrib?: number;
  form_contrib?: number;
  burst_contrib?: number;
}

export interface TokenData {
  text: string;
  score: number;        // expected in [0, 1]
  features: string[];   // e.g. ["lexical_complexity", "burstiness"]
  signed_score?: number;
}

export interface SentenceInfo {
  sentence_id: number;
  start: number;        // word index (inclusive)
  end: number;          // word index (exclusive)
  text: string;
  score: number;        // optional normalized [0, 1], if you use it later
}

interface AnalysisOutputProps {
  words: string[];
  spans: SpanData[];
  tokens?: TokenData[];         // optional, but preferred
  sentences?: SentenceInfo[];   // optional, for future sentence-level UI
  showUncertainty: boolean;
  sensitivity: number;          // 0–1 threshold
  prediction: number;           // 0 = human, 1 = AI
  showOpposing: boolean;
  wordSupportAI?: number[];
  wordSupportHuman?: number[];
  wordSaliencyAI?: number[];
  wordSaliencyHuman?: number[];
}

export const AnalysisOutput = ({
  words,
  spans,
  tokens,
  sentences,
  showUncertainty,
  sensitivity,
  prediction,
  showOpposing,
  wordSupportAI,
  wordSupportHuman,
  wordSaliencyAI,
  wordSaliencyHuman,
}: AnalysisOutputProps) => {
  const [hoveredSpan, setHoveredSpan] = useState<SpanData | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });

  // Map each word index → all spans that cover it
  const wordToSpans = new Map<number, SpanData[]>();
  spans.forEach((span) => {
    for (let i = span.start; i < span.end; i++) {
      if (!wordToSpans.has(i)) {
        wordToSpans.set(i, [span]);
      } else {
        wordToSpans.get(i)!.push(span);
      }
    }
  });

  // Pick the highest-scoring span at a given word (for tooltip & style)
  const getTopSpanAtWord = (index: number): SpanData | null => {
    const spansAtWord = wordToSpans.get(index);
    if (!spansAtWord || spansAtWord.length === 0) return null;

    return spansAtWord.reduce((best, current) =>
      current.score > best.score ? current : best
    );
  };

  const effectivePrediction = showOpposing ? (prediction === 1 ? 0 : 1) : prediction;
  const targetIsAI = effectivePrediction === 1;
  const highlightVar = targetIsAI ? "--ai-like" : "--human-like";
  const highlightColor = `hsl(var(${highlightVar}))`;

  const getWordImportance = (index: number): number => {
    // Use gradient-based saliency directly (class-specific)
    const saliency = targetIsAI ? wordSaliencyAI : wordSaliencyHuman;
    if (saliency && saliency[index] !== undefined) {
      return Math.max(0, Math.min(saliency[index] ?? 0, 1));
    }

    // Fallback: Use word support arrays
    const support = targetIsAI ? wordSupportAI : wordSupportHuman;
    if (support && support[index] !== undefined) {
      return Math.max(0, Math.min(support[index] ?? 0, 1));
    }

    // Last resort: Use token scores
    if (tokens && tokens[index]) {
      const fallbackScore = Math.max(0, Math.min(tokens[index].score ?? 0, 1));
      return fallbackScore;
    }

    return 0;
  };

  const getWordStyle = (index: number) => {
    // Handle whitespace-only words (no highlighting needed)
    const word = words[index];
    if (word && /^\s+$/.test(word)) {
      return {}; // No styling for whitespace
    }
    
    const importance = getWordImportance(index);
    const spacingOnly: CSSProperties = {
      marginRight: "0.35rem",
    };

    if (importance < sensitivity) {
      return spacingOnly;
    }

    const topSpan = getTopSpanAtWord(index);
    // Increase opacity for better visibility: scale from 0.3 to 0.85 based on importance
    const minOpacity = 0.3;
    const maxOpacity = 0.85;
    const opacity = minOpacity + (importance * (maxOpacity - minOpacity));

    const isBorderline =
      showUncertainty &&
      importance >= sensitivity &&
      importance < Math.min(sensitivity + 0.15, 1);

    const baseStyle: CSSProperties = {
      backgroundColor: `hsl(var(${highlightVar}) / ${opacity})`,
      color: "inherit", // Keep text white for visibility
      padding: "0.125rem 0.25rem",
      borderRadius: "0.25rem",
      cursor: topSpan ? "pointer" : "default",
      transition: "box-shadow 0.2s ease, background-color 0.2s ease",
      marginRight: "0.35rem",
      fontWeight: importance > 0.6 ? "500" : "normal",
    };

    if (isBorderline) {
      baseStyle.border = "1px dashed rgb(234, 179, 8)";
      baseStyle.backgroundColor = "rgba(234, 179, 8, 0.20)";
      baseStyle.boxShadow = "none";
    }

    return baseStyle;
  };

  const handleMouseEnter = (index: number, e: React.MouseEvent) => {
    const topSpan = getTopSpanAtWord(index);
    if (topSpan) {
      setHoveredSpan(topSpan);
      setTooltipPosition({ x: e.clientX, y: e.clientY });
    }
  };

  return (
    <div className="relative">
      <div className="prose prose-invert max-w-none text-foreground leading-relaxed whitespace-pre-wrap">
        {words.map((word, index) => {
          // Handle whitespace-only words (newlines, spaces, etc.)
          const isWhitespace = /^\s+$/.test(word);
          if (isWhitespace) {
            return <span key={index} className="whitespace-pre">{word}</span>;
          }
          
          return (
            <span
              key={index}
              className="inline-block hover:shadow-lg"
              style={getWordStyle(index)}
              onMouseEnter={(e) => handleMouseEnter(index, e)}
              onMouseLeave={() => setHoveredSpan(null)}
              onMouseMove={(e) =>
                setTooltipPosition({ x: e.clientX, y: e.clientY })
              }
            >
              {word}
            </span>
          );
        })}
      </div>


      {hoveredSpan && (
        <TokenTooltip
          span={hoveredSpan}
          position={tooltipPosition}
          prediction={effectivePrediction}
        />
      )}
    </div>
  );
};
