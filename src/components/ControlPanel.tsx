import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Download } from "lucide-react";

interface ControlPanelProps {
  sensitivity: number;
  onSensitivityChange: (value: number) => void;
  showUncertainty: boolean;
  onShowUncertaintyChange: (value: boolean) => void;
  onExport: () => void;
}

export const ControlPanel = ({
  sensitivity,
  onSensitivityChange,
  showUncertainty,
  onShowUncertaintyChange,
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
              Adjust the threshold for highlighting spans
            </p>
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
          <p className="font-semibold text-foreground mb-2">Highlighting Legend</p>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-ai-like" />
            <span>AI-generated spans</span>
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
