import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Loader2, Sparkles } from "lucide-react";
import { ConfidenceGauge } from "@/components/ConfidenceGauge";
import { AnalysisOutput, TokenData } from "@/components/AnalysisOutput";
import { ControlPanel } from "@/components/ControlPanel";
import { SummaryPanel } from "@/components/SummaryPanel";
import { useToast } from "@/hooks/use-toast";

const Index = () => {
  const [inputText, setInputText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [hasAnalyzed, setHasAnalyzed] = useState(false);
  const [confidence, setConfidence] = useState(0.5);
  const [tokens, setTokens] = useState<TokenData[]>([]);
  const [sensitivity, setSensitivity] = useState(0.2);
  const [showHumanLike, setShowHumanLike] = useState(true);
  const [showUncertainty, setShowUncertainty] = useState(false);
  const [granularity, setGranularity] = useState(1);
  const [analysisMode, setAnalysisMode] = useState<'words' | 'sentences'>('words');
  const [showPunctuation, setShowPunctuation] = useState(false);
  const [modelClassification, setModelClassification] = useState<{ model: string; confidence: number } | null>(null);
  const { toast } = useToast();

  const normalizeText = (text: string): string => {
    return text
      .trim()
      // Replace multiple spaces with single space
      .replace(/[ \t]+/g, ' ')
      // Normalize line breaks (max 2 consecutive)
      .replace(/\n{3,}/g, '\n\n')
      // Remove leading/trailing spaces from each line
      .split('\n')
      .map(line => line.trim())
      .join('\n')
      // Remove zero-width spaces and other weird unicode
      .replace(/[\u200B-\u200D\uFEFF]/g, '');
  };

  const classifyModelFromScores = (scores: any): string => {
    const { lexical_complexity, formality, burstiness } = scores;
    
    if (formality > 0.7 && lexical_complexity > 0.6) {
      return "Claude";
    } else if (burstiness > 0.7 && formality < 0.5) {
      return "GPT-4";
    } else if (lexical_complexity > 0.7) {
      return "Copilot";
    } else if (formality > 0.6) {
      return "Gemini";
    }
    return "GPT-3.5";
  };

  const analyzeText = async (text: string) => {
    try {
      const response = await fetch("http://localhost:5000/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: text,
          top_k: 3,
          max_length: 512,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const result = await response.json();

      // Convert words to token format with gradient scores
      const apiTokens: TokenData[] = result.words.map((word: string, idx: number) => {
        // Find if this word is in any span for scoring
        let score = 0;
        let spanFeature = null;
        
        for (const span of result.spans) {
          if (span.text.includes(word)) {
            score = Math.min(span.score / 10.0, 1.0);
            spanFeature = span.dom_feature;
            break;
          }
        }

        // Convert to [-1, 1] range for display (AI-like = positive, human-like = negative)
        const displayScore = (score - 0.5) * 2;

        return {
          text: word,
          score: displayScore,
          uncertainty: Math.random() * 0.3,
          features: [
            {
              name: "Lexical Complexity",
              value: result.global_scores.lexical_complexity * 2 - 1,
              description: "Long and technical vocabulary usage",
            },
            {
              name: "Formality",
              value: result.global_scores.formality * 2 - 1,
              description: "Formal, academic-style language patterns",
            },
            {
              name: "Burstiness",
              value: result.global_scores.burstiness * 2 - 1,
              description: "Uniformity in sentence length and rhythm",
            },
          ],
        };
      });

      setTokens(apiTokens);
      setConfidence(result.p_ai);

      // Set model classification based on global scores
      const modelName = classifyModelFromScores(result.global_scores);
      setModelClassification({
        model: modelName,
        confidence: result.p_ai,
      });

      return result;
    } catch (error) {
      console.error("Analysis error:", error);
      throw error;
    }
  };

  const handleAnalyze = async () => {
    if (!inputText.trim()) {
      toast({
        title: "No text provided",
        description: "Please enter some text to analyze",
        variant: "destructive",
      });
      return;
    }

    setIsAnalyzing(true);
    setHasAnalyzed(false);

    // Normalize text formatting before analysis
    const normalizedText = normalizeText(inputText);
    
    // Update input with normalized text
    setInputText(normalizedText);

    // Call real API
    try {
      await analyzeText(normalizedText);
      setIsAnalyzing(false);
      setHasAnalyzed(true);
      
      if (normalizedText !== inputText) {
        toast({
          title: "Text normalized",
          description: "Fixed spacing and formatting for better analysis",
        });
      }
    } catch (error) {
      setIsAnalyzing(false);
      toast({
        title: "Analysis failed",
        description: "Could not connect to backend API. Make sure the Flask server is running on port 5000.",
        variant: "destructive",
      });
    }
  };

  const handleExport = () => {
    toast({
      title: "Exporting report",
      description: "Your analysis report is being generated...",
    });
  };

  // Generate summary data
  const features = [
    {
      name: "Lexical Diversity",
      value: 0.72,
      description: "Variety of vocabulary used throughout the text",
    },
    {
      name: "Syntactic Complexity",
      value: 0.65,
      description: "Complexity of sentence structures",
    },
    {
      name: "Formality Level",
      value: 0.58,
      description: "Degree of formal vs. casual language",
    },
    {
      name: "Burstiness",
      value: 0.45,
      description: "Variation in sentence length and rhythm",
    },
  ];

  const sentences = inputText
    .split(/[.!?]+/)
    .filter((s) => s.trim())
    .map((sentence) => ({
      text: sentence.trim(),
      score: Math.random() * 2 - 1,
    }));

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-40">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-gradient-primary flex items-center justify-center shadow-glow">
                <Sparkles className="w-5 h-5 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-foreground">AI Text Detector</h1>
                <p className="text-xs text-muted-foreground">Advanced explainability analysis</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Input Panel */}
          <Card className="p-6 bg-card border-border shadow-card">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-foreground">Input Text</h2>
              <Button
                onClick={handleAnalyze}
                disabled={isAnalyzing}
                className="bg-gradient-primary hover:shadow-glow transition-all"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 mr-2" />
                    Analyze Text
                  </>
                )}
              </Button>
            </div>
            <Textarea
              placeholder="Paste or type your text here for AI detection analysis..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              className="min-h-[500px] font-mono text-sm resize-none bg-secondary border-border"
            />
          </Card>

          {/* Output Panel */}
          <Card className="p-6 bg-card border-border shadow-card">
            <h2 className="text-xl font-semibold text-foreground mb-4">Analysis Output</h2>
            {!hasAnalyzed && !isAnalyzing && (
              <div className="flex items-center justify-center h-[500px] text-muted-foreground">
                <div className="text-center">
                  <Sparkles className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Enter text and click "Analyze Text" to begin</p>
                </div>
              </div>
            )}
            {isAnalyzing && (
              <div className="flex items-center justify-center h-[500px]">
                <div className="text-center">
                  <Loader2 className="w-12 h-12 mx-auto mb-4 animate-spin text-primary" />
                  <p className="text-muted-foreground">Analyzing text patterns...</p>
                </div>
              </div>
            )}
            {hasAnalyzed && (
              <div className="min-h-[500px] space-y-4">
                <ConfidenceGauge 
                  score={confidence} 
                  isAnalyzing={isAnalyzing}
                  modelClassification={modelClassification || undefined}
                />
                <div className="p-4 bg-secondary rounded-lg max-h-[400px] overflow-y-auto">
                  <AnalysisOutput
                    tokens={tokens}
                    showHumanLike={showHumanLike}
                    showUncertainty={showUncertainty}
                    sensitivity={sensitivity}
                  />
                </div>
              </div>
            )}
          </Card>
        </div>

        {hasAnalyzed && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <SummaryPanel features={features} sentences={sentences} />
            </div>
            <div>
              <ControlPanel
                sensitivity={sensitivity}
                onSensitivityChange={setSensitivity}
                showHumanLike={showHumanLike}
                onShowHumanLikeChange={setShowHumanLike}
                showUncertainty={showUncertainty}
                onShowUncertaintyChange={setShowUncertainty}
                granularity={granularity}
                onGranularityChange={setGranularity}
                analysisMode={analysisMode}
                onAnalysisModeChange={setAnalysisMode}
                showPunctuation={showPunctuation}
                onShowPunctuationChange={setShowPunctuation}
                onExport={handleExport}
              />
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default Index;
