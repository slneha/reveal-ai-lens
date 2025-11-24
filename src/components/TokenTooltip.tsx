import { SpanData } from "./AnalysisOutput";
import { TrendingUp } from "lucide-react";

interface TokenTooltipProps {
  span: SpanData;
  position: { x: number; y: number };
}

export const TokenTooltip = ({ span, position }: TokenTooltipProps) => {
  const scorePercentage = Math.round(Math.min(span.score / 10.0, 1.0) * 100);

  const getFeatureLabel = (feature: string) => {
    switch (feature) {
      case "lexical_complexity":
        return "Lexical Complexity";
      case "formality":
        return "Formality";
      case "burstiness":
        return "Burstiness";
      default:
        return feature;
    }
  };

  return (
    <div
      className="fixed z-50 w-80 p-4 bg-popover border border-border rounded-lg shadow-card animate-scale-in"
      style={{
        left: `${position.x + 10}px`,
        top: `${position.y + 10}px`,
        pointerEvents: "none",
      }}
    >
      <div className="space-y-3">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="font-mono text-xs text-muted-foreground mb-1">Detected Span</div>
            <div className="font-semibold text-foreground break-words">"{span.text}"</div>
          </div>
          <TrendingUp className="w-5 h-5 text-ai-like flex-shrink-0 ml-2" />
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">AI Contribution:</span>
            <span className="font-semibold text-ai-like">
              {scorePercentage}%
            </span>
          </div>

          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Score:</span>
            <span className="font-medium">{span.score.toFixed(2)}</span>
          </div>
        </div>

        <div className="pt-2 border-t border-border">
          <div className="text-xs font-semibold text-muted-foreground mb-2">Dominant Feature</div>
          <div className="p-2 bg-secondary/50 rounded">
            <div className="font-medium text-sm text-foreground mb-1">
              {getFeatureLabel(span.dom_feature)}
            </div>
            <p className="text-xs text-muted-foreground">{span.reason}</p>
          </div>
        </div>

        {(span.lex_contrib !== undefined || span.form_contrib !== undefined || span.burst_contrib !== undefined) && (
          <div className="pt-2 border-t border-border">
            <div className="text-xs font-semibold text-muted-foreground mb-2">Feature Contributions:</div>
            <div className="space-y-1.5 text-xs">
              {span.lex_contrib !== undefined && (
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Lexical:</span>
                  <span className="font-medium">{span.lex_contrib.toFixed(3)}</span>
                </div>
              )}
              {span.form_contrib !== undefined && (
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Formality:</span>
                  <span className="font-medium">{span.form_contrib.toFixed(3)}</span>
                </div>
              )}
              {span.burst_contrib !== undefined && (
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Burstiness:</span>
                  <span className="font-medium">{span.burst_contrib.toFixed(3)}</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
