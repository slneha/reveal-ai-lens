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

  const classifyModel = (text: string) => {
    // Mock model classification based on text patterns
    const gptIndicators = text.match(/\b(furthermore|moreover|delve|comprehensive)\b/gi)?.length || 0;
    const claudeIndicators = text.match(/\b(certainly|i'd be happy|let me|i understand)\b/gi)?.length || 0;
    const copilotIndicators = text.match(/\b(code|function|implement|solution)\b/gi)?.length || 0;
    const geminiIndicators = text.match(/\b(explore|discover|innovative|creative)\b/gi)?.length || 0;

    const models = [
      { name: 'GPT-4', score: gptIndicators + Math.random() * 2 },
      { name: 'Claude', score: claudeIndicators + Math.random() * 2 },
      { name: 'GitHub Copilot', score: copilotIndicators + Math.random() * 2 },
      { name: 'Gemini', score: geminiIndicators + Math.random() * 2 },
    ];

    models.sort((a, b) => b.score - a.score);
    const totalScore = models.reduce((sum, m) => sum + m.score, 0);
    const confidence = totalScore > 0 ? models[0].score / totalScore : 0.5;

    return {
      model: models[0].name,
      confidence: Math.min(0.95, Math.max(0.3, confidence)),
    };
  };

  const mockAnalyze = (text: string) => {
    // Classify the model first
    const classification = classifyModel(text);
    setModelClassification(classification);

    // Simulate token-level analysis based on granularity
    let segments: string[];
    
    if (analysisMode === 'sentences') {
      segments = text.split(/([.!?]+\s+)/).filter(s => s.trim());
    } else if (granularity === 3) {
      // Sentence-level granularity
      segments = text.split(/([.!?]+)/).filter(s => s.trim());
    } else if (granularity === 2) {
      // Phrase-level granularity (split by commas and conjunctions)
      segments = text.split(/([,;]|\band\b|\bor\b|\bbut\b)/i).filter(s => s.trim());
    } else {
      // Word-level granularity
      segments = text.split(/(\s+)/);
    }

    const mockTokens: TokenData[] = segments.map((segment) => {
      const isAiIndicator = segment.toLowerCase().match(/\b(furthermore|moreover|consequently|thus|therefore|notably|delve)\b/);
      const isHumanIndicator = segment.toLowerCase().match(/\b(like|just|really|actually|basically|kinda)\b/);
      const hasPunctuation = showPunctuation && segment.match(/[—;:]/);
      
      let score = Math.random() * 2 - 1; // Random between -1 and 1
      
      if (isAiIndicator) score = Math.max(0.5, score);
      if (isHumanIndicator) score = Math.min(-0.5, score);
      if (hasPunctuation) score = Math.max(0.3, score); // Em dash and semicolons slightly AI-like

      return {
        text: segment,
        score,
        uncertainty: Math.random() * 0.5,
        features: [
          {
            name: "Formality Score",
            value: Math.random() * 2 - 1,
            description: isAiIndicator ? "Formal language pattern detected" : "Casual language pattern",
          },
          {
            name: "Sentence Complexity",
            value: Math.random() * 2 - 1,
            description: "Measures syntactic complexity and structure",
          },
          {
            name: "Burstiness",
            value: Math.random() * 2 - 1,
            description: "Variation in sentence length and rhythm",
          },
          ...(hasPunctuation ? [{
            name: "Punctuation Pattern",
            value: 0.4,
            description: "Complex punctuation usage detected (em dash, semicolon)",
          }] : []),
        ],
      };
    });

    // Calculate overall confidence (weighted average)
    const avgScore = mockTokens.reduce((sum, token) => sum + token.score, 0) / mockTokens.length;
    const normalizedConfidence = (avgScore + 1) / 2; // Convert from [-1, 1] to [0, 1]

    setTokens(mockTokens);
    setConfidence(normalizedConfidence);
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

    // Simulate API call (longer for classification + analysis)
    setTimeout(() => {
      mockAnalyze(normalizedText);
      setIsAnalyzing(false);
      setHasAnalyzed(true);
      
      if (normalizedText !== inputText) {
        toast({
          title: "Text normalized",
          description: "Fixed spacing and formatting for better analysis",
        });
      }
    }, 2500);
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
