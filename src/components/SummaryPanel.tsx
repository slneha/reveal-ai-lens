import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface FeatureImportance {
  name: string;
  value: number;
  description: string;
}

interface SentenceScore {
  text: string;
  score: number;
}

interface SummaryPanelProps {
  features: FeatureImportance[];
  sentences: SentenceScore[];
}

export const SummaryPanel = ({ features, sentences }: SummaryPanelProps) => {
  return (
    <div className="space-y-6 animate-fade-in">
      <Card className="p-6 bg-card border-border shadow-card">
        <h3 className="text-lg font-semibold text-foreground mb-4">Feature Importance</h3>
        <div className="space-y-4">
          {features.map((feature, idx) => (
            <div key={idx} className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-foreground">{feature.name}</span>
                <span className="text-sm text-muted-foreground">{Math.abs(feature.value).toFixed(2)}</span>
              </div>
              <Progress 
                value={Math.abs(feature.value) * 100} 
                className="h-2"
              />
              <p className="text-xs text-muted-foreground">{feature.description}</p>
            </div>
          ))}
        </div>
      </Card>

      <Card className="p-6 bg-card border-border shadow-card">
        <h3 className="text-lg font-semibold text-foreground mb-4">Sentence-Level Analysis</h3>
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {sentences.map((sentence, idx) => (
            <div key={idx} className="p-3 bg-secondary rounded-lg space-y-2">
              <p className="text-sm text-foreground">{sentence.text}</p>
              <div className="flex items-center gap-2">
                <div className="flex-1 h-2 bg-background rounded-full overflow-hidden">
                  <div
                    className="h-full transition-all"
                    style={{
                      width: `${Math.abs(sentence.score) * 100}%`,
                      backgroundColor: sentence.score > 0 
                        ? "hsl(var(--ai-like))" 
                        : "hsl(var(--human-like))",
                    }}
                  />
                </div>
                <span className={`text-xs font-medium ${
                  sentence.score > 0 ? "text-ai-like" : "text-human-like"
                }`}>
                  {Math.round(Math.abs(sentence.score) * 100)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};
