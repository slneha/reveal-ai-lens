// Text feature extraction utilities ported from explainable.py

export interface TextFeatures {
  lex_ttr: number;
  lex_avg_word_len: number;
  lex_long_word_ratio: number;
  syn_avg_sent_len: number;
  syn_comma_per_sent: number;
  form_contraction_ratio: number;
  form_first_person_ratio: number;
  form_connector_ratio: number;
  burstiness: number;
}

const FORMAL_CONNECTORS = [
  "furthermore", "moreover", "in addition", "in contrast",
  "therefore", "consequently", "thus", "hence", "overall",
  "in conclusion", "it is important to note", "the results suggest"
];

const FIRST_PERSON = new Set(["i", "we", "my", "our", "me", "us"]);

function simpleWordTokens(text: string): string[] {
  if (!text) return [];
  const trimmed = text.trim();
  if (!trimmed) return [];
  // includes contractions like "don't"
  const tokens = trimmed.match(/\b\w+'\w+|\w+\b/g) || [];
  return tokens;
}

function sentences(text: string): string[] {
  if (!text) return [];
  const sents = text.split(/[.!?]+/).map(s => s.trim()).filter(s => s);
  return sents;
}

function lexicalFeatures(text: string): Pick<TextFeatures, 'lex_ttr' | 'lex_avg_word_len' | 'lex_long_word_ratio'> {
  const tokens = simpleWordTokens(text);
  if (tokens.length === 0) {
    return { lex_ttr: 0, lex_avg_word_len: 0, lex_long_word_ratio: 0 };
  }

  const types = new Set(tokens);
  const ttr = types.size / tokens.length;

  const wordLengths = tokens.map(w => w.length);
  const avgLen = wordLengths.reduce((a, b) => a + b, 0) / wordLengths.length;

  const longWords = tokens.filter(w => w.length >= 7);
  const longRatio = longWords.length / tokens.length;

  return {
    lex_ttr: ttr,
    lex_avg_word_len: avgLen,
    lex_long_word_ratio: longRatio
  };
}

function syntacticFeatures(text: string): Pick<TextFeatures, 'syn_avg_sent_len' | 'syn_comma_per_sent'> {
  const sents = sentences(text);
  if (sents.length === 0) {
    return { syn_avg_sent_len: 0, syn_comma_per_sent: 0 };
  }

  const sentLengths: number[] = [];
  const commaCounts: number[] = [];

  for (const s of sents) {
    const tokens = simpleWordTokens(s);
    sentLengths.push(tokens.length);
    commaCounts.push((s.match(/,/g) || []).length);
  }

  const avgLen = sentLengths.reduce((a, b) => a + b, 0) / sentLengths.length;
  const avgComma = commaCounts.reduce((a, b) => a + b, 0) / commaCounts.length;

  return {
    syn_avg_sent_len: avgLen,
    syn_comma_per_sent: avgComma
  };
}

function formalityFeatures(text: string): Pick<TextFeatures, 'form_contraction_ratio' | 'form_first_person_ratio' | 'form_connector_ratio'> {
  if (!text || !text.trim()) {
    return { form_contraction_ratio: 0, form_first_person_ratio: 0, form_connector_ratio: 0 };
  }

  const lower = text.toLowerCase();
  const tokens = simpleWordTokens(text);
  if (tokens.length === 0) {
    return { form_contraction_ratio: 0, form_first_person_ratio: 0, form_connector_ratio: 0 };
  }

  // contractions
  const contractions = tokens.filter(t => t.includes("'"));
  const contractionRatio = contractions.length / tokens.length;

  // first-person pronouns
  const firstPerson = tokens.filter(t => FIRST_PERSON.has(t.toLowerCase()));
  const firstPersonRatio = firstPerson.length / tokens.length;

  // formal connectors
  const sents = sentences(text);
  let connectorCount = 0;
  if (sents.length > 0) {
    for (const phrase of FORMAL_CONNECTORS) {
      connectorCount += (lower.match(new RegExp(phrase, 'g')) || []).length;
    }
  }
  const connectorRatio = sents.length > 0 ? connectorCount / sents.length : 0;

  return {
    form_contraction_ratio: contractionRatio,
    form_first_person_ratio: firstPersonRatio,
    form_connector_ratio: connectorRatio
  };
}

function computeBurstiness(text: string): number {
  const sents = sentences(text);
  if (sents.length < 2) return 0;

  const lengths = sents.map(s => simpleWordTokens(s).length);
  const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
  if (mean === 0) return 0;

  const variance = lengths.reduce((sum, len) => sum + Math.pow(len - mean, 2), 0) / lengths.length;
  const std = Math.sqrt(variance);
  return std / mean;
}

export function extractTextFeatures(text: string): TextFeatures {
  const lex = lexicalFeatures(text);
  const syn = syntacticFeatures(text);
  const form = formalityFeatures(text);
  const burst = computeBurstiness(text);

  return {
    ...lex,
    ...syn,
    ...form,
    burstiness: burst
  };
}

// Feature metadata for display
export const FEATURE_METADATA: Record<string, { label: string; description: string }> = {
  lex_long_word_ratio: {
    label: "Lexical Diversity",
    description: "Share of unique / long words compared to the total vocabulary."
  },
  lex_avg_word_len: {
    label: "Average Word Length",
    description: "Longer average words typically correlate with technical language."
  },
  form_connector_ratio: {
    label: "Formal Connectors",
    description: "Usage of phrases such as 'therefore' or 'moreover' per sentence."
  },
  form_first_person_ratio: {
    label: "First-Person Usage",
    description: "Frequency of I/we/my pronouns relative to total tokens."
  },
  form_contraction_ratio: {
    label: "Contractions",
    description: "Ratio of words like \"don't\" or \"it's\" that imply informal tone."
  },
  syn_avg_sent_len: {
    label: "Sentence Length",
    description: "Average number of tokens per sentence."
  },
  syn_comma_per_sent: {
    label: "Comma Density",
    description: "Average comma usage per sentence, a proxy for clause complexity."
  },
  burstiness: {
    label: "Burstiness",
    description: "Variance in sentence lengths; higher burstiness is more human-like."
  }
};

// Baselines and correlations from Python
export const FEATURE_CORRELATIONS: Record<string, number> = {
  lex_long_word_ratio: 0.370235,
  lex_avg_word_len: 0.370001,
  form_first_person_ratio: 0.147474,
  form_contraction_ratio: 0.023141,
  syn_avg_sent_len: -0.04263,
  syn_comma_per_sent: -0.046974,
  burstiness: -0.379292
};

export const FEATURE_BASELINES: Record<string, number> = {
  lex_long_word_ratio: 0.24,
  lex_avg_word_len: 4.6,
  form_connector_ratio: 0.04,
  form_first_person_ratio: 0.10,
  form_contraction_ratio: 0.08,
  syn_avg_sent_len: 18.0,
  syn_comma_per_sent: 0.45,
  burstiness: 0.55
};

export const FEATURE_SCALES: Record<string, number> = {
  lex_long_word_ratio: 0.25,
  lex_avg_word_len: 3.0,
  form_connector_ratio: 0.15,
  form_first_person_ratio: 0.20,
  form_contraction_ratio: 0.20,
  syn_avg_sent_len: 15.0,
  syn_comma_per_sent: 1.0,
  burstiness: 0.45
};
