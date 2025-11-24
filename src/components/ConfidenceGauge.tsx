import { useEffect, useState } from "react";

interface ConfidenceGaugeProps {
  score: number; // 0-1 range, where 0 is human, 1 is AI (p_ai)
  pHuman?: number; // Optional explicit p_human value for accuracy
  isAnalyzing?: boolean;
}

export const ConfidenceGauge = ({ score, pHuman, isAnalyzing }: ConfidenceGaugeProps) => {
  const [animatedScore, setAnimatedScore] = useState(0);
  const [animatedHuman, setAnimatedHuman] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedScore(score);
      if (pHuman !== undefined) {
        setAnimatedHuman(pHuman);
      } else {
        setAnimatedHuman(1 - score);
      }
    }, 100);
    return () => clearTimeout(timer);
  }, [score, pHuman]);

  const clamp01 = (value: number) => Math.min(1, Math.max(0, value ?? 0));
  
  // Use explicit p_human if provided, otherwise calculate from p_ai
  const aiPercent = clamp01(animatedScore) * 100;
  const humanPercent = pHuman !== undefined 
    ? clamp01(animatedHuman) * 100 
    : 100 - aiPercent;
  
  // Ensure they sum to 100% and show at least 1 decimal place
  const total = aiPercent + humanPercent;
  const normalizedAI = total > 0 ? (aiPercent / total) * 100 : 0;
  const normalizedHuman = total > 0 ? (humanPercent / total) * 100 : 0;
  
  // Show 2 decimal places for better precision, but at least 1
  const aiDisplay = normalizedAI < 0.1 || normalizedAI > 99.9 
    ? normalizedAI.toFixed(2) 
    : normalizedAI.toFixed(1);
  const humanDisplay = normalizedHuman < 0.1 || normalizedHuman > 99.9
    ? normalizedHuman.toFixed(2)
    : normalizedHuman.toFixed(1);

  return (
    <div className="w-full space-y-4 animate-fade-in">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-lg font-semibold text-foreground">Detection Results</h3>
      </div>

      <div className="flex gap-4 mb-2">
        <div className="flex-1 text-center">
          <div className="text-3xl font-bold text-human-like mb-1">{humanDisplay}%</div>
          <div className="text-sm text-muted-foreground">Human</div>
        </div>
        <div className="flex-1 text-center">
          <div className="text-3xl font-bold text-ai-like mb-1">{aiDisplay}%</div>
          <div className="text-sm text-muted-foreground">AI</div>
        </div>
      </div>

      <div className="relative h-6 overflow-hidden rounded-full bg-secondary flex">
        <div
          className="h-full transition-all duration-1000 ease-out bg-human-like flex items-center justify-end pr-2"
          style={{ width: `${normalizedHuman}%` }}
        >
          {normalizedHuman > 5 && (
            <span className="text-xs font-semibold text-white">{humanDisplay}%</span>
          )}
        </div>
        <div
          className="h-full transition-all duration-1000 ease-out bg-ai-like flex items-center justify-start pl-2"
          style={{ width: `${normalizedAI}%` }}
        >
          {normalizedAI > 5 && (
            <span className="text-xs font-semibold text-white">{aiDisplay}%</span>
          )}
        </div>
        {isAnalyzing && (
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
        )}
      </div>
    </div>
  );
};
