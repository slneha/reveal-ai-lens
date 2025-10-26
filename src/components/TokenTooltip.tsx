import { TokenData } from "./AnalysisOutput";
import { TrendingUp, TrendingDown } from "lucide-react";

interface TokenTooltipProps {
  token: TokenData;
  position: { x: number; y: number };
}

export const TokenTooltip = ({ token, position }: TokenTooltipProps) => {
  const isAiLike = token.score > 0;
  const scorePercentage = Math.abs(Math.round(token.score * 100));

  return (
    <div
      className="fixed z-50 w-72 p-4 bg-popover border border-border rounded-lg shadow-card animate-scale-in"
      style={{
        left: `${position.x + 10}px`,
        top: `${position.y + 10}px`,
        pointerEvents: "none",
      }}
    >
      <div className="space-y-3">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="font-mono text-sm text-muted-foreground mb-1">Token Analysis</div>
            <div className="font-semibold text-foreground break-words">"{token.text}"</div>
          </div>
          {isAiLike ? (
            <TrendingUp className="w-5 h-5 text-ai-like flex-shrink-0 ml-2" />
          ) : (
            <TrendingDown className="w-5 h-5 text-human-like flex-shrink-0 ml-2" />
          )}
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Likelihood:</span>
            <span className={`font-semibold ${isAiLike ? "text-ai-like" : "text-human-like"}`}>
              {scorePercentage}% {isAiLike ? "AI" : "Human"}
            </span>
          </div>

          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Uncertainty:</span>
            <span className="font-medium">±{Math.round(token.uncertainty * 100)}%</span>
          </div>
        </div>

        <div className="pt-2 border-t border-border">
          <div className="text-xs font-semibold text-muted-foreground mb-2">Top Contributing Features:</div>
          <div className="space-y-2">
            {token.features.slice(0, 3).map((feature, idx) => (
              <div key={idx} className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-foreground">{feature.name}</span>
                  <span className="text-xs text-muted-foreground">{Math.abs(feature.value).toFixed(2)}</span>
                </div>
                <div className="h-1 bg-secondary rounded-full overflow-hidden">
                  <div
                    className={`h-full ${feature.value > 0 ? "bg-ai-like" : "bg-human-like"}`}
                    style={{ width: `${Math.abs(feature.value) * 100}%` }}
                  />
                </div>
                <p className="text-xs text-muted-foreground">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>

        {token.uncertainty > 0.3 && (
          <div className="pt-2 border-t border-border">
            <div className="flex items-start gap-2 text-xs text-muted-foreground">
              <span className="text-yellow-500">⚠️</span>
              <span>High uncertainty detected. This prediction may be less reliable.</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
