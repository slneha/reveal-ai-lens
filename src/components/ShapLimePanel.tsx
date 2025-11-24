import { Card } from "@/components/ui/card";

interface ShapValue {
  word: string;
  shap_value: number;
  contribution: "AI" | "Human";
}

interface LimeValue {
  word: string;
  lime_score: number;
  contribution: "AI" | "Human";
}

interface ShapLimePanelProps {
  shap?: {
    shap_values?: ShapValue[];
    base_value?: number;
    error?: string;
  };
  lime?: {
    lime_scores?: LimeValue[];
    prediction?: number[];
    error?: string;
  };
}

export const ShapLimePanel = ({ shap, lime }: ShapLimePanelProps) => {
  const shapValues = shap?.shap_values || [];
  const limeScores = lime?.lime_scores || [];

  // Sort by absolute value for display
  const sortedShap = [...shapValues].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));
  const sortedLime = [...limeScores].sort((a, b) => Math.abs(b.lime_score) - Math.abs(a.lime_score));

  // Get max absolute values for normalization
  const maxShap = Math.max(...sortedShap.map(s => Math.abs(s.shap_value)), 1);
  const maxLime = Math.max(...sortedLime.map(l => Math.abs(l.lime_score)), 1);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* SHAP Visualization */}
      <Card className="p-6 bg-card border-border shadow-card">
        <h3 className="text-lg font-semibold text-foreground mb-4">
          SHAP Explanations
          {shap?.base_value !== undefined && (
            <span className="text-sm font-normal text-muted-foreground ml-2">
              (Base: {(shap.base_value * 100).toFixed(1)}%)
            </span>
          )}
        </h3>
        {shap?.error ? (
          <p className="text-sm text-muted-foreground">{shap.error}</p>
        ) : sortedShap.length === 0 ? (
          <p className="text-sm text-muted-foreground">No SHAP values available</p>
        ) : (
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {sortedShap.slice(0, 20).map((item, idx) => {
              const isAI = item.contribution === "AI";
              const normalizedValue = Math.abs(item.shap_value) / maxShap;
              const width = Math.min(normalizedValue * 100, 100);
              
              return (
                <div key={idx} className="space-y-1">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium text-foreground">{item.word}</span>
                    <span className={`text-xs font-semibold ${
                      isAI ? "text-ai-like" : "text-human-like"
                    }`}>
                      {item.shap_value >= 0 ? "+" : ""}
                      {item.shap_value.toFixed(4)}
                    </span>
                  </div>
                  <div className="h-2 bg-background rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        isAI ? "bg-ai-like" : "bg-human-like"
                      }`}
                      style={{ width: `${width}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Card>

      {/* LIME Visualization */}
      <Card className="p-6 bg-card border-border shadow-card">
        <h3 className="text-lg font-semibold text-foreground mb-4">
          LIME Explanations
          {lime?.prediction && (
            <span className="text-sm font-normal text-muted-foreground ml-2">
              (Human: {(lime.prediction[0] * 100).toFixed(1)}%, AI: {(lime.prediction[1] * 100).toFixed(1)}%)
            </span>
          )}
        </h3>
        {lime?.error ? (
          <p className="text-sm text-muted-foreground">{lime.error}</p>
        ) : sortedLime.length === 0 ? (
          <p className="text-sm text-muted-foreground">No LIME scores available</p>
        ) : (
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {sortedLime.slice(0, 20).map((item, idx) => {
              const isAI = item.contribution === "AI";
              const normalizedValue = Math.abs(item.lime_score) / maxLime;
              const width = Math.min(normalizedValue * 100, 100);
              
              return (
                <div key={idx} className="space-y-1">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium text-foreground">{item.word}</span>
                    <span className={`text-xs font-semibold ${
                      isAI ? "text-ai-like" : "text-human-like"
                    }`}>
                      {item.lime_score >= 0 ? "+" : ""}
                      {item.lime_score.toFixed(4)}
                    </span>
                  </div>
                  <div className="h-2 bg-background rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        isAI ? "bg-ai-like" : "bg-human-like"
                      }`}
                      style={{ width: `${width}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Card>
    </div>
  );
};

