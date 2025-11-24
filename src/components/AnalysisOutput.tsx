import { useState } from "react";
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

interface AnalysisOutputProps {
  words: string[];
  spans: SpanData[];
  showUncertainty: boolean;
  sensitivity: number;
}

export const AnalysisOutput = ({ 
  words,
  spans,
  showUncertainty,
  sensitivity 
}: AnalysisOutputProps) => {
  const [hoveredSpan, setHoveredSpan] = useState<SpanData | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });

  // Create a map of word indices to their spans
  const wordToSpan = new Map<number, SpanData>();
  spans.forEach(span => {
    for (let i = span.start; i < span.end; i++) {
      if (!wordToSpan.has(i) || wordToSpan.get(i)!.score < span.score) {
        wordToSpan.set(i, span);
      }
    }
  });

  const getWordStyle = (index: number) => {
    const span = wordToSpan.get(index);
    if (!span) return {};

    // Normalize score (0-10 range to 0-1)
    const normalizedScore = Math.min(span.score / 10.0, 1.0);
    
    if (normalizedScore < sensitivity) {
      return {};
    }

    const opacity = Math.min(normalizedScore * 0.4, 0.6);
    
    return {
      backgroundColor: `hsl(var(--ai-like) / ${opacity})`,
      color: normalizedScore > 0.5 ? "hsl(var(--ai-like))" : "inherit",
      padding: "0.125rem 0.25rem",
      borderRadius: "0.25rem",
      cursor: "pointer",
    };
  };

  const handleMouseEnter = (index: number, e: React.MouseEvent) => {
    const span = wordToSpan.get(index);
    if (span) {
      setHoveredSpan(span);
      setTooltipPosition({ x: e.clientX, y: e.clientY });
    }
  };

  return (
    <div className="relative">
      <div className="prose prose-invert max-w-none text-foreground leading-relaxed">
        {words.map((word, index) => (
          <span
            key={index}
            className="inline-block transition-all duration-200 hover:shadow-lg"
            style={getWordStyle(index)}
            onMouseEnter={(e) => handleMouseEnter(index, e)}
            onMouseLeave={() => setHoveredSpan(null)}
            onMouseMove={(e) => setTooltipPosition({ x: e.clientX, y: e.clientY })}
          >
            {word}{" "}
          </span>
        ))}
      </div>

      {hoveredSpan && (
        <TokenTooltip
          span={hoveredSpan}
          position={tooltipPosition}
        />
      )}
    </div>
  );
};
