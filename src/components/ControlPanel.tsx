import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Download } from "lucide-react";

interface ControlPanelProps {
  sensitivity: number;
  onSensitivityChange: (value: number) => void;
  showHumanLike: boolean;
  onShowHumanLikeChange: (value: boolean) => void;
  showUncertainty: boolean;
  onShowUncertaintyChange: (value: boolean) => void;
  granularity: number;
  onGranularityChange: (value: number) => void;
  analysisMode: 'words' | 'sentences';
  onAnalysisModeChange: (value: 'words' | 'sentences') => void;
  showPunctuation: boolean;
  onShowPunctuationChange: (value: boolean) => void;
  onExport: () => void;
}

export const ControlPanel = ({
  sensitivity,
  onSensitivityChange,
  showHumanLike,
  onShowHumanLikeChange,
  showUncertainty,
  onShowUncertaintyChange,
  granularity,
  onGranularityChange,
  analysisMode,
  onAnalysisModeChange,
  showPunctuation,
  onShowPunctuationChange,
  onExport,
}: ControlPanelProps) => {
  return (
    <div className="space-y-6 p-6 bg-card border border-border rounded-lg shadow-card">
      <div>
        <h3 className="text-lg font-semibold text-foreground mb-4">Display Controls</h3>
        
        <div className="space-y-6">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="sensitivity" className="text-sm font-medium">
                Highlight Sensitivity
              </Label>
              <span className="text-sm text-muted-foreground">{Math.round(sensitivity * 100)}%</span>
            </div>
            <Slider
              id="sensitivity"
              min={0}
              max={1}
              step={0.05}
              value={[sensitivity]}
              onValueChange={(values) => onSensitivityChange(values[0])}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              Adjust the threshold for highlighting tokens
            </p>
          </div>

          <div className="flex items-center justify-between space-x-2">
            <Label htmlFor="show-human" className="text-sm font-medium cursor-pointer">
              Show human-like tokens
            </Label>
            <Switch
              id="show-human"
              checked={showHumanLike}
              onCheckedChange={onShowHumanLikeChange}
            />
          </div>

          <div className="flex items-center justify-between space-x-2">
            <Label htmlFor="show-uncertainty" className="text-sm font-medium cursor-pointer">
              Show uncertainty overlay
            </Label>
            <Switch
              id="show-uncertainty"
              checked={showUncertainty}
              onCheckedChange={onShowUncertaintyChange}
            />
          </div>

          <div className="flex items-center justify-between space-x-2">
            <Label htmlFor="show-punctuation" className="text-sm font-medium cursor-pointer">
              Highlight punctuation patterns
            </Label>
            <Switch
              id="show-punctuation"
              checked={showPunctuation}
              onCheckedChange={onShowPunctuationChange}
            />
          </div>

          <div className="space-y-3 pt-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="granularity" className="text-sm font-medium">
                Analysis Granularity
              </Label>
              <span className="text-sm text-muted-foreground">
                {granularity === 1 ? 'Words' : granularity === 2 ? 'Phrases' : 'Sentences'}
              </span>
            </div>
            <Slider
              id="granularity"
              min={1}
              max={3}
              step={1}
              value={[granularity]}
              onValueChange={(values) => onGranularityChange(values[0])}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              Adjust token grouping level
            </p>
          </div>

          <div className="space-y-2">
            <Label className="text-sm font-medium">Analysis Mode</Label>
            <div className="flex gap-2">
              <Button
                variant={analysisMode === 'words' ? 'default' : 'outline'}
                size="sm"
                onClick={() => onAnalysisModeChange('words')}
                className="flex-1"
              >
                Words
              </Button>
              <Button
                variant={analysisMode === 'sentences' ? 'default' : 'outline'}
                size="sm"
                onClick={() => onAnalysisModeChange('sentences')}
                className="flex-1"
              >
                Sentences
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="pt-4 border-t border-border">
        <Button onClick={onExport} className="w-full" variant="outline">
          <Download className="w-4 h-4 mr-2" />
          Export Analysis Report
        </Button>
      </div>

      <div className="pt-4 border-t border-border">
        <div className="text-xs text-muted-foreground space-y-2">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-ai-like" />
            <span>AI-like tokens</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-human-like" />
            <span>Human-like tokens</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded border border-dashed border-yellow-500 bg-yellow-500/20" />
            <span>High uncertainty</span>
          </div>
        </div>
      </div>
    </div>
  );
};
