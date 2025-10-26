import { useEffect, useState } from "react";

interface ConfidenceGaugeProps {
  score: number; // 0-1 range, where 0 is human, 1 is AI
  isAnalyzing?: boolean;
  modelClassification?: {
    model: string;
    confidence: number;
  };
}

export const ConfidenceGauge = ({ score, isAnalyzing, modelClassification }: ConfidenceGaugeProps) => {
  const [animatedScore, setAnimatedScore] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedScore(score);
    }, 100);
    return () => clearTimeout(timer);
  }, [score]);

  const percentage = Math.round(score * 100);
  const isAiLikely = score > 0.5;
  
  const getGradientStop = () => {
    return `${percentage}%`;
  };

  return (
    <div className="w-full space-y-4 animate-fade-in">
      {modelClassification && (
        <div className="p-4 bg-secondary/50 rounded-lg border border-border">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-foreground">Detected Model</h3>
            <span className="text-xs text-muted-foreground">
              {Math.round(modelClassification.confidence * 100)}% confidence
            </span>
          </div>
          <div className="text-2xl font-bold bg-gradient-primary bg-clip-text text-transparent">
            {modelClassification.model}
          </div>
        </div>
      )}
      
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-foreground">AI Detection Confidence</h3>
        <div className="flex items-center gap-2">
          <span className={`text-2xl font-bold ${isAiLikely ? "text-ai-like" : "text-human-like"}`}>
            {percentage}%
          </span>
          <span className="text-sm text-muted-foreground">
            {isAiLikely ? "AI-like" : "Human-like"}
          </span>
        </div>
      </div>

      <div className="relative h-4 overflow-hidden rounded-full bg-secondary">
        <div
          className="h-full transition-all duration-1000 ease-out"
          style={{
            width: getGradientStop(),
            background: "linear-gradient(90deg, hsl(var(--human-like)), hsl(60 100% 50%), hsl(var(--ai-like)))",
            backgroundSize: "200% 100%",
            backgroundPosition: `${100 - percentage}% 0`,
          }}
        />
        {isAnalyzing && (
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
        )}
      </div>

      <div className="flex justify-between text-xs text-muted-foreground">
        <span className="flex items-center gap-1">
          <div className="h-2 w-2 rounded-full bg-human-like" />
          Human-like
        </span>
        <span className="flex items-center gap-1">
          <div className="h-2 w-2 rounded-full bg-neutral" />
          Neutral
        </span>
        <span className="flex items-center gap-1">
          <div className="h-2 w-2 rounded-full bg-ai-like" />
          AI-like
        </span>
      </div>
    </div>
  );
};
