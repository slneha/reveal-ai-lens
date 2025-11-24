import { useEffect, useState } from "react";

interface ConfidenceGaugeProps {
  score: number; // 0-1 range, where 0 is human, 1 is AI
  isAnalyzing?: boolean;
}

export const ConfidenceGauge = ({ score, isAnalyzing }: ConfidenceGaugeProps) => {
  const [animatedScore, setAnimatedScore] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedScore(score);
    }, 100);
    return () => clearTimeout(timer);
  }, [score]);

  const aiPercentage = Math.round(score * 100);
  const humanPercentage = 100 - aiPercentage;

  return (
    <div className="w-full space-y-4 animate-fade-in">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-lg font-semibold text-foreground">Detection Results</h3>
      </div>

      <div className="flex gap-4 mb-2">
        <div className="flex-1 text-center">
          <div className="text-3xl font-bold text-human-like mb-1">{humanPercentage}%</div>
          <div className="text-sm text-muted-foreground">Human</div>
        </div>
        <div className="flex-1 text-center">
          <div className="text-3xl font-bold text-ai-like mb-1">{aiPercentage}%</div>
          <div className="text-sm text-muted-foreground">AI</div>
        </div>
      </div>

      <div className="relative h-6 overflow-hidden rounded-full bg-secondary flex">
        <div
          className="h-full transition-all duration-1000 ease-out bg-human-like flex items-center justify-end pr-2"
          style={{ width: `${humanPercentage}%` }}
        >
          {humanPercentage > 10 && (
            <span className="text-xs font-semibold text-white">{humanPercentage}%</span>
          )}
        </div>
        <div
          className="h-full transition-all duration-1000 ease-out bg-ai-like flex items-center justify-start pl-2"
          style={{ width: `${aiPercentage}%` }}
        >
          {aiPercentage > 10 && (
            <span className="text-xs font-semibold text-white">{aiPercentage}%</span>
          )}
        </div>
        {isAnalyzing && (
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
        )}
      </div>
    </div>
  );
};
