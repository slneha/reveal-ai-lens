import { useState } from "react";
import { TokenTooltip } from "./TokenTooltip";

export interface TokenData {
  text: string;
  score: number; // -1 to 1, where -1 is human-like, 1 is AI-like
  features: {
    name: string;
    value: number;
    description: string;
  }[];
  uncertainty: number;
}

interface AnalysisOutputProps {
  tokens: TokenData[];
  showHumanLike: boolean;
  showUncertainty: boolean;
  sensitivity: number;
}

export const AnalysisOutput = ({ 
  tokens, 
  showHumanLike, 
  showUncertainty,
  sensitivity 
}: AnalysisOutputProps) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });

  const getTokenColor = (token: TokenData) => {
    const normalizedScore = Math.abs(token.score);
    
    if (normalizedScore < sensitivity) {
      return "text-neutral";
    }

    if (token.score > 0) {
      // AI-like (red spectrum)
      const opacity = Math.min(normalizedScore * 1.2, 1);
      return `text-ai-like`;
    } else {
      // Human-like (green spectrum)
      if (!showHumanLike) return "text-foreground";
      const opacity = Math.min(normalizedScore * 1.2, 1);
      return `text-human-like`;
    }
  };

  const getBackgroundStyle = (token: TokenData) => {
    const normalizedScore = Math.abs(token.score);
    
    if (normalizedScore < sensitivity) {
      return {};
    }

    const opacity = Math.min(normalizedScore * 0.3, 0.5);
    
    if (showUncertainty && token.uncertainty > 0.3) {
      return {
        backgroundColor: `hsl(45 100% 50% / ${opacity * 0.5})`,
        border: "1px dashed hsl(45 100% 50% / 0.5)",
      };
    }

    if (token.score > 0) {
      return {
        backgroundColor: `hsl(var(--ai-like) / ${opacity})`,
      };
    } else if (showHumanLike) {
      return {
        backgroundColor: `hsl(var(--human-like) / ${opacity})`,
      };
    }

    return {};
  };

  const handleMouseEnter = (index: number, e: React.MouseEvent) => {
    setHoveredIndex(index);
    setTooltipPosition({ x: e.clientX, y: e.clientY });
  };

  return (
    <div className="relative">
      <div className="prose prose-invert max-w-none text-foreground leading-relaxed">
        {tokens.map((token, index) => (
          <span
            key={index}
            className={`${getTokenColor(token)} transition-all duration-200 cursor-pointer rounded px-1 py-0.5 hover:shadow-lg relative inline-block`}
            style={getBackgroundStyle(token)}
            onMouseEnter={(e) => handleMouseEnter(index, e)}
            onMouseLeave={() => setHoveredIndex(null)}
            onMouseMove={(e) => setTooltipPosition({ x: e.clientX, y: e.clientY })}
          >
            {token.text}
          </span>
        ))}
      </div>

      {hoveredIndex !== null && (
        <TokenTooltip
          token={tokens[hoveredIndex]}
          position={tooltipPosition}
        />
      )}
    </div>
  );
};
