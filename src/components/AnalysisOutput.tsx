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
}

export const AnalysisOutput = ({
  words,
  spans,
  tokens,
  sentences,
  showUncertainty,
  sensitivity,
  prediction,
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

  // Get importance score for a word, preferring tokens[] if provided
  const getWordImportance = (index: number): number => {
    if (tokens && tokens[index]) {
      const s = tokens[index].score;
      // ensure it's clamped to [0, 1]
      return Math.max(0, Math.min(s, 1));
    }

    // Fallback: infer from spans if tokens not available
    const topSpan = getTopSpanAtWord(index);
    if (!topSpan) return 0;

    // If your span scores are unnormalized, lightly normalize
    const raw = topSpan.score;
    const normalized = raw <= 1 ? raw : raw / 10.0;
    return Math.max(0, Math.min(normalized, 1));
  };

  const isAI = prediction === 1;
  const highlightVar = isAI ? "--ai-like" : "--human-like";
  const highlightColor = `hsl(var(${highlightVar}))`;

  const getWordStyle = (index: number) => {
    const importance = getWordImportance(index);
    const spacingOnly: CSSProperties = {
      marginRight: "0.35rem",
    };

    if (importance < sensitivity) {
      // Below threshold → keep spacing so words don't collapse together
      return spacingOnly;
    }

    const topSpan = getTopSpanAtWord(index);

    // Base AI-like highlight: stronger importance → deeper color
    const opacity = Math.min(importance * 0.45, 0.7);

    // Optional uncertainty accent: words just above threshold
    // are considered "borderline" and get a yellow hint when toggled on

    
    const isBorderline =
      showUncertainty &&
      importance >= sensitivity &&
      importance < Math.min(sensitivity + 0.15, 1);

    const baseStyle: CSSProperties = {
      backgroundColor: `hsl(var(${highlightVar}) / ${opacity})`,
      color: importance > 0.7 ? highlightColor : "inherit",
      padding: "0.125rem 0.25rem",
      borderRadius: "0.25rem",
      cursor: topSpan ? "pointer" : "default",
      transition: "box-shadow 0.2s ease, background-color 0.2s ease",
      marginRight: "0.35rem",
    };

    if (isBorderline) {
      // Match legend: border-dashed border-yellow-500 bg-yellow-500/20
      baseStyle.border = "1px dashed rgb(234, 179, 8)";             // border-yellow-500
      baseStyle.backgroundColor = "rgba(234, 179, 8, 0.20)";        // bg-yellow-500/20
      baseStyle.boxShadow = "none";                                 // remove AI glow
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
      <div className="prose prose-invert max-w-none text-foreground leading-relaxed">
        {words.map((word, index) => (
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
            {word}{" "}
          </span>
        ))}
      </div>


      {hoveredSpan && (
        <TokenTooltip
          span={hoveredSpan}
          position={tooltipPosition}
          prediction={prediction}
        />
      )}
    </div>
  );
};
