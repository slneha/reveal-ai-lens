# explainable.py
"""
Explainability utilities for AI-vs-Human text detector using
andreas122001/roberta-mixed-detector.

This module assumes:
- Binary labels: 0 = human-produced, 1 = machine-generated.
- You can compute per-text features (lexical, formality, burstiness)
  and use them together with gradient-based saliency to generate
  phrase-level explanations.

Typical workflow in a notebook:
    from explainable import (
        load_detector,
        add_text_features,
        init_feature_stats,
        explain_row_with_features,
    )

    tokenizer, model, device = load_detector()

    # df_train should at least have column "text"
    df_train = add_text_features(df_train, text_col="text")
    init_feature_stats(df_train)

    info = explain_row_with_features(df_train, row_idx=10, top_k=3)
"""

import math
import re
from collections import Counter
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from sklearn.linear_model import LinearRegression, Ridge
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, falling back to correlation-based weights")


try:
    import nltk
    from nltk.corpus import stopwords
    # Download stopwords if not already available
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    STOP = set(stopwords.words("english"))
except ImportError:
    # Fallback if nltk is not available
    STOP = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
        "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "through", "during", "before", "after",
        "above", "below", "up", "down", "in", "out", "on", "off", "over", "under", "again",
        "further", "then", "once"
    }


# ============================================================
# 0. MODEL LOADING (CPU by default)
# ============================================================

MODEL_NAME = "andreas122001/roberta-mixed-detector"

tokenizer = None  # will be set by load_detector()
model = None
device = torch.device("cpu")


def load_detector(model_name: str = MODEL_NAME, use_gpu: bool = False):
    """
    Load the RoBERTa-based AI detector.

    Parameters
    ----------
    model_name : str
        Hugging Face model id.
    use_gpu : bool
        If True and CUDA is available, put model on GPU.

    Returns
    -------
    tokenizer, model, device
    """
    global tokenizer, model, device

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()
    return tokenizer, model, device


# ============================================================
# 1. BASIC TEXT FEATURES (lexical / syntactic / formality)
# ============================================================

# --- helper for word tokenization ---


def _simple_word_tokens(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []
    # includes contractions like "don't"
    tokens = re.findall(r"\b\w+'\w+|\w+\b", text)
    return tokens


def _sentences(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    # very simple sentence splitting
    sents = re.split(r"[.!?]+", text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents


# --- lexical features: TTR, avg word length, long-word ratio ---


def lexical_features(text: str) -> pd.Series:
    tokens = _simple_word_tokens(text)
    if len(tokens) == 0:
        return pd.Series(
            {
                "lex_ttr": 0.0,
                "lex_avg_word_len": 0.0,
                "lex_long_word_ratio": 0.0,
            }
        )

    types = set(tokens)
    ttr = len(types) / len(tokens)

    word_lengths = [len(w) for w in tokens]
    avg_len = float(np.mean(word_lengths))

    long_words = [w for w in tokens if len(w) >= 7]  # threshold 7 chars
    long_ratio = len(long_words) / len(tokens)

    return pd.Series(
        {
            "lex_ttr": ttr,
            "lex_avg_word_len": avg_len,
            "lex_long_word_ratio": long_ratio,
        }
    )


# --- syntactic features: avg sentence length, commas per sentence ---


def syntactic_features(text: str) -> pd.Series:
    sents = _sentences(text)
    if len(sents) == 0:
        return pd.Series(
            {
                "syn_avg_sent_len": 0.0,
                "syn_comma_per_sent": 0.0,
            }
        )

    sent_lengths = []
    comma_counts = []

    for s in sents:
        tokens = _simple_word_tokens(s)
        sent_lengths.append(len(tokens))
        comma_counts.append(s.count(","))

    avg_len = float(np.mean(sent_lengths))
    avg_comma = float(np.mean(comma_counts))

    return pd.Series(
        {
            "syn_avg_sent_len": avg_len,
            "syn_comma_per_sent": avg_comma,
        }
    )


# --- formality features: connectors, first-person, contractions ---


FORMAL_CONNECTORS = {
    "furthermore",
    "moreover",
    "in addition",
    "in contrast",
    "therefore",
    "consequently",
    "thus",
    "hence",
    "overall",
    "in conclusion",
    "it is important to note",
    "the results suggest",
}

FIRST_PERSON = {"i", "we", "my", "our", "me", "us"}


def formality_features(text: str) -> pd.Series:
    if not isinstance(text, str) or not text.strip():
        return pd.Series(
            {
                "form_contraction_ratio": 0.0,
                "form_first_person_ratio": 0.0,
                "form_connector_ratio": 0.0,
            }
        )

    lower = text.lower()
    tokens = _simple_word_tokens(text)
    if len(tokens) == 0:
        return pd.Series(
            {
                "form_contraction_ratio": 0.0,
                "form_first_person_ratio": 0.0,
                "form_connector_ratio": 0.0,
            }
        )

    # contractions: simple heuristic
    contractions = [t for t in tokens if "'" in t]
    contraction_ratio = len(contractions) / len(tokens)

    # first-person pronouns
    first_person = [t.lower() for t in tokens if t.lower() in FIRST_PERSON]
    first_person_ratio = len(first_person) / len(tokens)

    # formal connectors by substring per sentence
    sentences = _sentences(text)
    if len(sentences) == 0:
        connector_ratio = 0.0
    else:
        connector_count = 0
        for phrase in FORMAL_CONNECTORS:
            connector_count += lower.count(phrase)
        connector_ratio = connector_count / len(sentences)

    return pd.Series(
        {
            "form_contraction_ratio": contraction_ratio,
            "form_first_person_ratio": first_person_ratio,
            "form_connector_ratio": connector_ratio,
        }
    )


# --- burstiness: sentence-length std / mean (lower = more AI-like) ---


def compute_burstiness(text: str) -> float:
    sents = _sentences(text)
    if len(sents) < 2:
        return 0.0
    lengths = [len(_simple_word_tokens(s)) for s in sents]
    if np.mean(lengths) == 0:
        return 0.0
    return float(np.std(lengths) / np.mean(lengths))


# --- stopword ratio ---


def stopword_ratio(text: str) -> float:
    """
    Measures the ratio of stopwords to total tokens.
    Higher stopword ratio may indicate more natural, human-like text.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    tokens = re.findall(r"\b\w+\b", text.lower())
    if len(tokens) == 0:
        return 0.0
    return float(sum(t in STOP for t in tokens) / len(tokens))


# --- semantic redundancy / repetition ---


def compute_redundancy(text: str) -> float:
    """
    Measures semantic redundancy/repetition.
    AI text tends to repeat similar phrases and concepts.
    Returns ratio of repeated bigrams/trigrams.
    """
    tokens = _simple_word_tokens(text.lower())
    if len(tokens) < 3:
        return 0.0
    
    # Count bigrams and trigrams
    bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
    trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
    
    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)
    
    # Ratio of repeated n-grams (appearing more than once)
    repeated_bigrams = sum(1 for count in bigram_counts.values() if count > 1)
    repeated_trigrams = sum(1 for count in trigram_counts.values() if count > 1)
    
    total_bigrams = len(bigrams)
    total_trigrams = len(trigrams)
    
    if total_bigrams == 0 and total_trigrams == 0:
        return 0.0
    
    bigram_redundancy = repeated_bigrams / total_bigrams if total_bigrams > 0 else 0.0
    trigram_redundancy = repeated_trigrams / total_trigrams if total_trigrams > 0 else 0.0
    
    return float((bigram_redundancy + trigram_redundancy) / 2.0)


# --- topic uniformity ---


def compute_topic_uniformity(text: str) -> float:
    """
    Measures how uniform the topic is across sentences.
    AI text tends to stay on topic, human text may drift.
    Uses TF-IDF-like measure of vocabulary consistency.
    """
    sents = _sentences(text)
    if len(sents) < 2:
        return 0.0
    
    sent_tokens = [_simple_word_tokens(s.lower()) for s in sents]
    all_tokens = set()
    for tokens in sent_tokens:
        all_tokens.update(tokens)
    
    if len(all_tokens) == 0:
        return 0.0
    
    # For each sentence, compute overlap with overall vocabulary
    overlaps = []
    for tokens in sent_tokens:
        if len(tokens) == 0:
            continue
        sent_set = set(tokens)
        overlap = len(sent_set & all_tokens) / len(sent_set) if len(sent_set) > 0 else 0.0
        overlaps.append(overlap)
    
    if not overlaps:
        return 0.0
    
    return float(np.mean(overlaps))


# --- hedge vs assertive polarity ---


ASSERTIVE_PHRASES = {
    "is designed to", "is intended to", "ensures", "provides", "guarantees",
    "will", "must", "always", "never", "certainly", "definitely", "clearly",
    "obviously", "undoubtedly", "absolutely", "completely", "entirely"
}

HEDGE_PHRASES = {
    "might", "may", "could", "possibly", "perhaps", "maybe", "tends to",
    "can sometimes", "appears to", "seems to", "suggests", "indicates",
    "likely", "probably", "somewhat", "rather", "quite", "fairly"
}


def compute_assertiveness(text: str) -> pd.Series:
    """
    Measures hedge vs assertive language.
    AI tends to be more assertive, humans mix certainty and doubt.
    Returns assertive_ratio and hedge_ratio.
    """
    if not isinstance(text, str) or not text.strip():
        return pd.Series({"assertive_ratio": 0.0, "hedge_ratio": 0.0})
    
    lower = text.lower()
    tokens = _simple_word_tokens(text)
    if len(tokens) == 0:
        return pd.Series({"assertive_ratio": 0.0, "hedge_ratio": 0.0})
    
    sentences = _sentences(text)
    if len(sentences) == 0:
        return pd.Series({"assertive_ratio": 0.0, "hedge_ratio": 0.0})
    
    assertive_count = 0
    hedge_count = 0
    
    # Count phrase matches
    for phrase in ASSERTIVE_PHRASES:
        assertive_count += lower.count(phrase)
    
    for phrase in HEDGE_PHRASES:
        hedge_count += lower.count(phrase)
    
    # Also check individual assertive words in tokens
    assertive_words = {"will", "must", "always", "never", "certainly", "definitely"}
    hedge_words = {"might", "may", "could", "possibly", "perhaps", "maybe", "likely", "probably"}
    
    for token in tokens:
        token_lower = token.lower()
        if token_lower in assertive_words:
            assertive_count += 1
        elif token_lower in hedge_words:
            hedge_count += 1
    
    assertive_ratio = assertive_count / len(sentences) if len(sentences) > 0 else 0.0
    hedge_ratio = hedge_count / len(sentences) if len(sentences) > 0 else 0.0
    
    # Net assertiveness: positive = more assertive (AI-like), negative = more hedged (human-like)
    net_assertiveness = assertive_ratio - hedge_ratio
    
    return pd.Series({
        "assertive_ratio": assertive_ratio,
        "hedge_ratio": hedge_ratio,
        "net_assertiveness": net_assertiveness,
    })


# --- chain-of-clarification patterns ---


CLARIFICATION_PATTERNS = {
    "this means that", "in other words", "to put it simply", "that is to say",
    "in other terms", "put differently", "more specifically", "in simpler terms",
    "to clarify", "to explain", "to elaborate", "to be more precise"
}


def compute_clarification_patterns(text: str) -> float:
    """
    Measures use of chain-of-clarification patterns.
    AI often adds auto-explanations like "This means that...", "In other words..."
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    
    lower = text.lower()
    sentences = _sentences(text)
    if len(sentences) == 0:
        return 0.0
    
    clarification_count = 0
    for pattern in CLARIFICATION_PATTERNS:
        clarification_count += lower.count(pattern)
    
    return float(clarification_count / len(sentences)) if len(sentences) > 0 else 0.0


# --- meta-writing phrases ---


META_WRITING_PHRASES = {
    "this paper", "this study", "this research", "this article", "this work",
    "the results demonstrate", "the findings show", "the analysis reveals",
    "the goal of this", "the purpose of this", "the aim of this",
    "we investigate", "we examine", "we analyze", "we explore",
    "in this paper", "in this study", "in this research", "in this article",
    "the paper", "the study", "the research", "the article"
}


def compute_meta_writing(text: str) -> float:
    """
    Measures use of meta-writing phrases.
    AI frequently refers to its own writing task, especially in academic contexts.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    
    lower = text.lower()
    sentences = _sentences(text)
    if len(sentences) == 0:
        return 0.0
    
    meta_count = 0
    for phrase in META_WRITING_PHRASES:
        meta_count += lower.count(phrase)
    
    return float(meta_count / len(sentences)) if len(sentences) > 0 else 0.0


# --- wrapper: add all text-level features to a DataFrame ---


def add_text_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Adds feature columns to df inplace and also returns df.

    Requires column `text_col` (default "text").
    """
    lex = df[text_col].apply(lexical_features)
    syn = df[text_col].apply(syntactic_features)
    form = df[text_col].apply(formality_features)
    burst = df[text_col].apply(lambda t: compute_burstiness(t))

    df = df.copy()
    df[["lex_ttr", "lex_avg_word_len", "lex_long_word_ratio"]] = lex
    df[["syn_avg_sent_len", "syn_comma_per_sent"]] = syn
    df[
        ["form_contraction_ratio", "form_first_person_ratio", "form_connector_ratio"]
    ] = form
    df["burstiness"] = burst

    return df


# ============================================================
# 2. GLOBAL FEATURE STATS (for z-scoring)
# ============================================================

FEATURE_STATS_COLS = [
    "lex_avg_word_len",
    "lex_long_word_ratio",
    "form_connector_ratio",
    "form_first_person_ratio",
    "burstiness",
]

FEATURE_STATS: Dict[str, Dict[str, float]] = {}


FEATURE_CORRELATIONS: Dict[str, float] = {
    "lex_long_word_ratio": 0.370235,
    "lex_avg_word_len": 0.370001,
    "form_first_person_ratio": 0.147474,
    "form_contraction_ratio": 0.023141,
    "syn_avg_sent_len": -0.04263,
    "syn_comma_per_sent": -0.046974,
    "burstiness": -0.379292,
    # New features (estimated correlations - adjust based on your data)
    "topic_uniformity": 0.05,  # High uniformity → AI (low impact factor, not displayed separately)
    "net_assertiveness": 0.15,  # High assertiveness → AI
    "stopword_ratio": 0.191680,  # Higher stopword ratio → AI
}


def compute_regression_weights(
    normalized_features: Dict[str, float],
    p_ai: float,
    feature_groups: Dict[str, List[str]],
) -> Dict[str, float]:
    """
    Compute feature weights using regression-based approach.
    Given p_ai and normalized feature values, compute weights that explain the prediction.
    
    Args:
        normalized_features: Dictionary of normalized feature values [0, 1]
        p_ai: Probability of AI (target variable)
        feature_groups: Dictionary mapping group names to feature keys
        
    Returns:
        Dictionary of weights for each feature group
    """
    if not SKLEARN_AVAILABLE:
        # Fallback to correlation-based weights
        weights = {}
        for group_name, feature_keys in feature_groups.items():
            weights[group_name] = float(np.mean([
                FEATURE_CORRELATIONS.get(key, 0.0) for key in feature_keys
            ]))
        return weights
    
    # Prepare feature matrix: aggregate features by group
    # Since we only have one data point, we can't do proper regression
    # Instead, we'll use correlations as weights but scale them based on feature values
    group_names = []
    feature_values = []
    
    for group_name, feature_keys in feature_groups.items():
        # Average normalized features in this group
        group_value = float(np.mean([
            normalized_features.get(key, 0.5) for key in feature_keys
        ]))
        group_names.append(group_name)
        feature_values.append(group_value)
    
    if len(group_names) == 0:
        # Fallback
        return {name: FEATURE_CORRELATIONS.get(name, 0.0) for name in feature_groups.keys()}
    
    # Use correlations as base weights, but adjust based on how much each feature
    # contributes to explaining the current p_ai value
    # Simple approach: use correlations directly, but scale by feature deviation from neutral
    weights = {}
    prior_weights = {}
    
    for i, group_name in enumerate(group_names):
        # Get correlation-based weight
        corr_weight = float(np.mean([
            FEATURE_CORRELATIONS.get(key, 0.0) for key in feature_groups[group_name]
        ]))
        prior_weights[group_name] = corr_weight
        
        # Adjust weight based on feature value and p_ai
        # If feature is high and p_ai is high, increase weight; if feature is low and p_ai is low, increase weight
        feature_val = feature_values[i]
        feature_deviation = (feature_val - 0.5) * 2.0  # -1 to 1
        p_ai_deviation = (p_ai - 0.5) * 2.0  # -1 to 1
        
        # If feature deviation aligns with p_ai deviation, increase weight
        # Otherwise, decrease weight
        alignment = feature_deviation * p_ai_deviation * corr_weight
        adjusted_weight = corr_weight + alignment * 0.3  # Small adjustment
        
        weights[group_name] = float(adjusted_weight)
    
    return weights

FEATURE_BASELINES: Dict[str, float] = {
    "lex_long_word_ratio": 0.24,
    "lex_avg_word_len": 4.6,
    "form_connector_ratio": 0.04,
    "form_first_person_ratio": 0.10,
    "form_contraction_ratio": 0.08,
    "syn_avg_sent_len": 18.0,
    "syn_comma_per_sent": 0.45,
    "burstiness": 0.55,
}

FEATURE_SCALES: Dict[str, float] = {
    "lex_long_word_ratio": 0.25,
    "lex_avg_word_len": 3.0,
    "form_connector_ratio": 0.15,
    "form_first_person_ratio": 0.20,
    "form_contraction_ratio": 0.20,
    "syn_avg_sent_len": 15.0,
    "syn_comma_per_sent": 1.0,
    "burstiness": 0.45,
}

FEATURE_METADATA: Dict[str, Dict[str, str]] = {
    "lex_long_word_ratio": {
        "label": "Lexical Diversity",
        "description": "Share of unique / long words compared to the total vocabulary.",
    },
    "lex_avg_word_len": {
        "label": "Average Word Length",
        "description": "Longer average words typically correlate with technical language.",
    },
    "form_connector_ratio": {
        "label": "Formal Connectors",
        "description": "Usage of phrases such as 'therefore' or 'moreover' per sentence.",
    },
    "form_first_person_ratio": {
        "label": "First-Person Usage",
        "description": "Frequency of I/we/my pronouns relative to total tokens.",
    },
    "form_contraction_ratio": {
        "label": "Contractions",
        "description": "Ratio of words like \"don't\" or \"it's\" that imply informal tone.",
    },
    "syn_avg_sent_len": {
        "label": "Sentence Length",
        "description": "Average number of tokens per sentence.",
    },
    "syn_comma_per_sent": {
        "label": "Comma Density",
        "description": "Average comma usage per sentence, a proxy for clause complexity.",
    },
    "burstiness": {
        "label": "Burstiness",
        "description": "Variance in sentence lengths; higher burstiness is more human-like.",
    },
    "topic_uniformity": {
        "label": "Topic Uniformity",
        "description": "Consistency of vocabulary across sentences. AI tends to stay on topic.",
    },
    "net_assertiveness": {
        "label": "Assertiveness vs Hedging",
        "description": "Balance of assertive phrases (AI-like) vs hedging phrases (human-like).",
    },
    "stopword_ratio": {
        "label": "Stopword Ratio",
        "description": "Ratio of common stopwords to total tokens. Higher ratio may indicate AI-generated text.",
    },
}


def init_feature_stats(df: pd.DataFrame):
    """
    Initialize global mean / std for key feature columns, to be used
    for global scoring.

    Call this once after you've computed features on your training
    set (df_train).
    """
    global FEATURE_STATS
    FEATURE_STATS = {}
    for col in FEATURE_STATS_COLS:
        mu = float(df[col].mean())
        sd = float(df[col].std() or 1.0)
        FEATURE_STATS[col] = {"mean": mu, "std": sd}


def _z_to_01(z: float) -> float:
    """Squash z-score into (0,1) with logistic."""
    return float(1.0 / (1.0 + math.exp(-z)))


def _norm_feature(col: str, value: float) -> float:
    info = FEATURE_STATS.get(col, {"mean": 0.0, "std": 1.0})
    mu = info["mean"]
    sd = info["std"] or 1.0
    z = (value - mu) / sd
    return _z_to_01(z)


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _center01(x: float) -> float:
    return float(max(-1.0, min(1.0, (x - 0.5) * 2.0)))


def _feature_deviation(key: str, value: float) -> float:
    baseline = FEATURE_BASELINES.get(key, 0.0)
    scale = FEATURE_SCALES.get(key, 1.0) or 1.0
    raw = (value - baseline) / scale
    return float(max(-1.0, min(1.0, raw)))


def get_global_feature_scores_for_row(df: pd.DataFrame, row_idx: int) -> Dict[str, float]:
    """
    Using already-computed numeric features in df, build 3 high-level scores in [0,1]:
      - lexical_complexity
      - formality
      - burstiness (AI-ness from low burstiness)

    df must have the feature columns and FEATURE_STATS must be initialized.
    """
    row = df.iloc[row_idx]

    # lexical: combine long-word ratio + avg word length
    s_len = _norm_feature("lex_avg_word_len", row["lex_avg_word_len"])
    s_long = _norm_feature("lex_long_word_ratio", row["lex_long_word_ratio"])
    lexical_complexity = 0.5 * (s_len + s_long)

    # formality: more formal connectors, fewer first-person pronouns
    s_conn = _norm_feature("form_connector_ratio", row["form_connector_ratio"])
    s_fp = _norm_feature("form_first_person_ratio", row["form_first_person_ratio"])
    formality = 0.7 * s_conn + 0.3 * (1.0 - s_fp)

    # burstiness: low burstiness (uniform sentence lengths) correlated with AI
    b_raw = float(row["burstiness"])
    info_b = FEATURE_STATS.get("burstiness", {"mean": 0.0, "std": 1.0})
    b_mu = info_b["mean"]
    b_sd = info_b["std"] or 1.0
    z_b = (b_raw - b_mu) / b_sd
    burstiness_score = _z_to_01(-z_b)  # flip sign: low burstiness → high AI-score

    return {
        "lexical_complexity": float(lexical_complexity),
        "formality": float(formality),
        "burstiness": float(burstiness_score),
    }


# ============================================================
# 3. GRADIENT-BASED TOKEN IMPORTANCE
# ============================================================


def get_token_importance(
    text: str,
    target_class: int,
    max_length: int = 256,
) -> Tuple[List[str], np.ndarray, List[int]]:
    """
    Returns tokens, importance scores, and token IDs for a single text.
    target_class: 0 (human-produced) or 1 (machine-generated).

    Uses gradients wrt input embeddings as saliency.
    """
    assert tokenizer is not None and model is not None, "Call load_detector() first."

    # Tokenize
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = enc["input_ids"].to(device)  # [1, L]
    attention_mask = enc["attention_mask"].to(device)

    # Get input embeddings
    emb_layer = model.get_input_embeddings()
    input_embeds = emb_layer(input_ids)  # [1, L, H]
    input_embeds.retain_grad()

    # Forward with embeds
    model.zero_grad()
    outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
    logits = outputs.logits  # [1, 2]
    logit = logits[0, target_class]

    # Backward
    logit.backward()

    # Gradients wrt embeddings
    grads = input_embeds.grad[0]  # [L, H]
    token_importance = grads.norm(dim=-1).detach().cpu().numpy()

    token_ids = input_ids[0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    return tokens, token_importance, token_ids


# ============================================================
# 4. TOKENS → WORDS + SENTENCE IDS
# ============================================================


def roberta_tokens_to_words_and_map(tokens: List[str], token_ids: List[int] = None) -> Tuple[List[str], List[List[int]]]:
    """
    Turn RoBERTa tokens into words AND keep mapping.
    Preserves whitespace, newlines, and punctuation by using tokenizer decode.
    Returns:
      words   : list[str]
      mapping : list[list[int]]  (each word -> list of token indices)
    """
    if tokenizer is not None and token_ids is not None:
        # Use tokenizer decode to get properly formatted text with whitespace preserved
        try:
            decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
            
            # Split into words while preserving following whitespace (including newlines)
            import re
            # Match word followed by whitespace, or just whitespace at start/end
            # This preserves newlines, multiple spaces, etc.
            word_pattern = r'\S+\s*|\s+'
            word_matches = re.findall(word_pattern, decoded_text)
            
            words = []
            mapping = []
            token_idx = 0
            specials = {"<s>", "</s>"}
            
            # Align decoded words with tokens
            for word_with_space in word_matches:
                word_clean = word_with_space.strip()
                if not word_clean:  # Pure whitespace
                    # Include whitespace with previous word if exists, or as separate entry
                    if words:
                        words[-1] += word_with_space
                    else:
                        words.append(word_with_space)
                        mapping.append([])
                    continue
                
                # Find tokens that make up this word
                word_tokens = []
                accumulated = ""
                
                while token_idx < len(tokens):
                    tok = tokens[token_idx]
                    if tok in specials:
                        token_idx += 1
                        continue
                    
                    # Remove Ġ prefix (space marker)
                    tok_clean = tok[1:] if tok.startswith("Ġ") else tok
                    
                    # Normalize for comparison (remove extra spaces)
                    test_accumulated = (accumulated + tok_clean).replace(" ", "")
                    word_normalized = word_clean.replace(" ", "")
                    
                    if word_normalized.startswith(test_accumulated) or test_accumulated == word_normalized:
                        word_tokens.append(token_idx)
                        accumulated += tok_clean
                        token_idx += 1
                        if test_accumulated == word_normalized:
                            break
                    elif accumulated and not word_normalized.startswith(test_accumulated):
                        # No match, break
                        break
                    else:
                        word_tokens.append(token_idx)
                        accumulated += tok_clean
                        token_idx += 1
                        if test_accumulated == word_normalized:
                            break
                
                if word_tokens:
                    # Include trailing whitespace with the word
                    words.append(word_with_space)
                    mapping.append(word_tokens)
            
            if words:
                return words, mapping
        except Exception as e:
            # Fall back to original method if decode fails
            print(f"Warning: Tokenizer decode failed, using fallback: {e}")
            pass
    
    # Fallback: original method
    words: List[str] = []
    mapping: List[List[int]] = []

    current_word = ""
    current_tok_idxs: List[int] = []

    specials = {"<s>", "</s>"}

    for i, tok in enumerate(tokens):
        if tok in specials:
            continue

        if tok.startswith("Ġ"):  # new word
            if current_word:
                words.append(current_word)
                mapping.append(current_tok_idxs)
            current_word = tok[1:]
            current_tok_idxs = [i]
        else:
            current_word += tok
            current_tok_idxs.append(i)

    if current_word:
        words.append(current_word)
        mapping.append(current_tok_idxs)

    return words, mapping


def assign_sentence_ids(words: List[str]) -> List[int]:
    """
    Very lightweight sentence segmentation:
    increments sentence id when we see ., !, or ? at the end of a word.
    Returns: list[int] of same length as words.
    """
    sent_ids = []
    sid = 0
    for w in words:
        sent_ids.append(sid)
        if re.search(r"[.!?]\s*$", w):
            sid += 1
    return sent_ids


# ============================================================
# 5. WORD-LEVEL FEATURE CONTRIBUTIONS
# ============================================================

# build a simple vocab of connector *words* from FORMAL_CONNECTORS
CONNECTOR_WORDS = set()
for phrase in FORMAL_CONNECTORS:
    for w in phrase.split():
        CONNECTOR_WORDS.add(w.lower())


def get_word_level_feature_scores(
    words: List[str], sent_ids: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each word, return:
      lex_scores   – higher for longer words
      form_scores  – higher for formal connectors / academic phrases
      burst_scores – higher for sentences with more uniform length
    """
    n = len(words)
    if n == 0:
        return (
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
        )

    lower_words = [w.lower() for w in words]

    # lexical: normalized word length
    lengths = np.array([len(w) for w in words], dtype=float)
    max_len = float(lengths.max()) if n > 0 else 1.0
    if max_len <= 0:
        lex_scores = np.zeros_like(lengths)
        length_factor = np.ones_like(lengths)
    else:
        lex_scores = lengths / max_len
        length_factor = np.clip(0.35 + 0.65 * (lengths / max_len), 0.35, 1.0)

    # formality: connectors, assertive phrases, clarification patterns, meta-writing
    form_scores = np.zeros(n, dtype=float)
    
    # Build context windows for phrase detection
    text_lower = " ".join(lower_words).lower()
    
    for i, w in enumerate(lower_words):
        # Formal connectors
        if w in CONNECTOR_WORDS:
            form_scores[i] += 1.0
        if w in {"thus", "therefore", "consequently", "furthermore", "moreover"}:
            form_scores[i] += 1.0
        if w in {"we", "our"}:
            form_scores[i] += 0.3
        
        # Assertive phrases (AI-like)
        assertive_words = {"will", "must", "always", "never", "certainly", "definitely", "ensures", "provides", "guarantees"}
        if w in assertive_words:
            form_scores[i] += 1.5

    if form_scores.max() > 0:
        form_scores = form_scores / form_scores.max()

    # burstiness: compute per-sentence score = 1 - normalized deviation from mean length
    sent_lengths: Dict[int, int] = {}
    for s in sent_ids:
        sent_lengths[s] = sent_lengths.get(s, 0) + 1

    if len(sent_lengths) <= 1:
        burst_scores = np.ones(n, dtype=float)
    else:
        lens = np.array(list(sent_lengths.values()), dtype=float)
        mean_len = lens.mean()
        max_dev = max(1.0, np.max(np.abs(lens - mean_len)))
        sent_score = {
            s: 1.0 - abs(L - mean_len) / max_dev for s, L in sent_lengths.items()
        }
        burst_scores = np.array([sent_score[s] for s in sent_ids], dtype=float)

    if burst_scores.max() > 0:
        burst_scores = burst_scores / burst_scores.max()
    burst_scores = burst_scores * length_factor
    if burst_scores.max() > 0:
        burst_scores = burst_scores / burst_scores.max()

    return lex_scores, form_scores, burst_scores


# ============================================================
# 6. SPAN SELECTION + EXPLANATIONS
# ============================================================


def select_top_spans(
    words: List[str],
    total_score: np.ndarray,
    lex_contrib: np.ndarray,
    form_contrib: np.ndarray,
    burst_contrib: np.ndarray,
    top_k: int = 3,
    min_len: int = 3,
    max_len: int = 8,
    target_label: str = "ai",
) -> List[Dict[str, Any]]:
    """
    Greedy selection of top-k non-overlapping spans based on total_score
    (over words, not tokens).
    Returns a list of dicts:
      { 'start', 'end', 'text', 'score', 'dom_feature', 'reason' }
    """
    L = len(words)
    if L == 0:
        return []

    spans: List[Dict[str, Any]] = []

    # generate all candidate spans
    for n in range(min_len, max_len + 1):
        if L < n:
            continue
        for i in range(L - n + 1):
            j = i + n
            s = float(total_score[i:j].sum())
            spans.append({"start": i, "end": j, "score": s})

    if not spans:
        return []

    # sort by score
    spans.sort(key=lambda d: d["score"], reverse=True)

    chosen: List[Dict[str, Any]] = []
    used = np.zeros(L, dtype=bool)

    for sp in spans:
        if any(used[sp["start"] : sp["end"]]):
            continue
        chosen.append(sp)
        used[sp["start"] : sp["end"]] = True
        if len(chosen) >= top_k:
            break

    # attach text + dominant feature + reason
    out: List[Dict[str, Any]] = []
    for sp in chosen:
        i, j = sp["start"], sp["end"]
        span_words = words[i:j]

        span_lex = float(lex_contrib[i:j].mean())
        span_form = float(form_contrib[i:j].mean())
        span_burst = float(burst_contrib[i:j].mean())

        feat_vals = {
            "lexical_complexity": span_lex,
            "formality": span_form,
            "burstiness": span_burst,
        }
        dom_feat = max(feat_vals, key=feat_vals.get)

        if target_label == "ai":
            if dom_feat == "lexical_complexity":
                reason = "Technical or repeated vocabulary drives the AI decision."
            elif dom_feat == "formality":
                reason = "Formal connectors / impersonal tone increase AI likelihood."
            else:
                reason = "Uniform sentence rhythms look machine-generated."
        else:
            if dom_feat == "lexical_complexity":
                reason = "Conversational vocabulary tempers the AI likelihood."
            elif dom_feat == "formality":
                reason = "Informal voice and personal cues feel human-written."
            else:
                reason = "Highly varied sentence lengths point toward human authorship."

        out.append(
            {
                "start": i,
                "end": j,
                "text": " ".join(span_words),
                "score": float(sp["score"]),
                "dom_feature": dom_feat,
                "reason": reason,
                "lex_contrib": span_lex,
                "form_contrib": span_form,
                "burst_contrib": span_burst,
            }
        )

    return out


# ============================================================
# 7. MAIN ENTRYPOINTS
# ============================================================


def explain_row_with_features(
    df: pd.DataFrame,
    row_idx: int,
    max_length: int = 512,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Full pipeline for a single df row:
      - uses df's features and FEATURE_STATS to get 3 global scores
      - gets gradient-based token importance from RoBERTa detector
      - converts to words
      - builds word-level feature contributions
      - returns top spans + explanations

    df is expected to have:
      - column "text"
      - feature columns (lex_..., form_..., burstiness)
      - optionally "pred" and "p_ai" for the classifier outputs
    """
    row = df.iloc[row_idx]
    text = row["text"]

    # if you want to explain "true" class, use row["label"]
    # if you want to explain model decision, use row["pred"]
    target_class = int(row.get("pred", row.get("label", 1)))  # default 1

    # 1) global (document-level) feature scores in [0,1]
    global_scores = get_global_feature_scores_for_row(df, row_idx)

    # 2) gradient-based token saliency
    tokens, tok_scores, token_ids = get_token_importance(text, target_class, max_length=max_length)
    tok_scores = np.array(tok_scores, dtype=float)
    if tok_scores.max() > 0:
        tok_scores = tok_scores / tok_scores.max()

    # 3) tokens → words
    words, mapping = roberta_tokens_to_words_and_map(tokens, token_ids)
    if len(words) == 0:
        return {
            "prediction": target_class,
            "true_label": int(row.get("label", -1)),
            "p_ai": float(row.get("p_ai", np.nan)),
            "global_scores": global_scores,
            "spans": [],
            "metadata": row.get("Metadata", ""),
        }

    # aggregate token saliency at word level
    word_sal = np.zeros(len(words), dtype=float)
    for i, idxs in enumerate(mapping):
        idxs = [k for k in idxs if k < len(tok_scores)]
        if not idxs:
            continue
        word_sal[i] = float(tok_scores[idxs].mean())
    if word_sal.max() > 0:
        word_sal = word_sal / word_sal.max()

    # 4) sentence ids & local feature scores
    sent_ids = assign_sentence_ids(words)
    lex_loc, form_loc, burst_loc = get_word_level_feature_scores(words, sent_ids)

    def _nz_norm(x: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype=float)
        if x.max() > 0:
            x = x / x.max()
        return x

    lex_loc = _nz_norm(lex_loc)
    form_loc = _nz_norm(form_loc)
    burst_loc = _nz_norm(burst_loc)

    # 5) combine: global strength × local indicator × gradient saliency
    g_lex = global_scores["lexical_complexity"]
    g_form = global_scores["formality"]
    g_burst = global_scores["burstiness"]

    lex_contrib = g_lex * lex_loc * word_sal
    form_contrib = g_form * form_loc * word_sal
    burst_contrib = g_burst * burst_loc * word_sal

    total_score = lex_contrib + form_contrib + burst_contrib

    spans = select_top_spans(
        words,
        total_score,
        lex_contrib,
        form_contrib,
        burst_contrib,
        top_k=top_k,
        min_len=3,
        max_len=8,
    )

    return {
        "prediction": target_class,
        "true_label": int(row.get("label", -1)),
        "p_ai": float(row.get("p_ai", np.nan)),
        "global_scores": global_scores,
        "spans": spans,
        "metadata": row.get("Metadata", ""),
        "words": words,
    }


def explain_text_with_features(
    text: str,
    max_length: int = 512,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Convenience function for a *single raw text*, without df_train.
    It:
      - runs the detector
      - computes features on the fly for this text only
      - uses simple heuristics (no global z-scores) to create global scores

    This is handy for a Lovable / Flask UI where you just get one text.

    NOTE: Global scores here are *relative to this document*, not to your
    full dataset. For your paper, continue using explain_row_with_features
    on df_train with init_feature_stats().
    """
    # 1) run detector
    assert tokenizer is not None and model is not None, "Call load_detector() first."

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits[0].cpu().numpy()

    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    p_human, p_ai = float(probs[0]), float(probs[1])
    pred = int(np.argmax(probs))  # 0=Human,1=AI

    # 2) compute features for this text only
    lex = lexical_features(text)
    form = formality_features(text)
    syn = syntactic_features(text)
    burst = compute_burstiness(text)
    topic_uniformity = compute_topic_uniformity(text)
    assertiveness = compute_assertiveness(text)
    stopword_ratio_val = stopword_ratio(text)

    raw_feature_values = {
        "lex_long_word_ratio": float(lex["lex_long_word_ratio"]),
        "lex_avg_word_len": float(lex["lex_avg_word_len"]),
        "form_connector_ratio": float(form["form_connector_ratio"]),
        "form_first_person_ratio": float(form["form_first_person_ratio"]),
        "form_contraction_ratio": float(form["form_contraction_ratio"]),
        "syn_avg_sent_len": float(syn["syn_avg_sent_len"]),
        "syn_comma_per_sent": float(syn["syn_comma_per_sent"]),
        "burstiness": float(burst),
        "topic_uniformity": float(topic_uniformity),
        "net_assertiveness": float(assertiveness["net_assertiveness"]),
        "stopword_ratio": float(stopword_ratio_val),
    }

    normalized_features = {
        "lex_long_word_ratio": _clamp01(raw_feature_values["lex_long_word_ratio"]),
        "lex_avg_word_len": _clamp01((raw_feature_values["lex_avg_word_len"] - 3.0) / 7.0),
        "form_connector_ratio": _clamp01(raw_feature_values["form_connector_ratio"]),
        "form_first_person_ratio": _clamp01(raw_feature_values["form_first_person_ratio"]),
        "form_contraction_ratio": _clamp01(raw_feature_values["form_contraction_ratio"]),
        "syn_avg_sent_len": _clamp01(raw_feature_values["syn_avg_sent_len"] / 40.0),
        "syn_comma_per_sent": _clamp01(raw_feature_values["syn_comma_per_sent"] / 5.0),
        "burstiness": _clamp01(math.tanh(raw_feature_values["burstiness"])),
        "topic_uniformity": _clamp01(raw_feature_values["topic_uniformity"]),
        "net_assertiveness": _clamp01((raw_feature_values["net_assertiveness"] + 1.0) / 2.0),  # Map [-1,1] to [0,1]
        "stopword_ratio": _clamp01(raw_feature_values["stopword_ratio"]),
    }

    feature_impacts: List[Dict[str, Any]] = []
    for key, corr in FEATURE_CORRELATIONS.items():
        if key == "burstiness":
            # Local burstiness evidence already covers this signal to avoid duplication
            continue
        meta = FEATURE_METADATA.get(
            key,
            {
                "label": key.replace("_", " ").title(),
                "description": "",
            },
        )
        raw_val = raw_feature_values.get(key, 0.0)
        deviation = _feature_deviation(key, raw_val)
        signed_score = corr * deviation
        feature_impacts.append(
            {
                "key": key,
                "label": meta["label"],
                "description": meta["description"],
                "raw_value": raw_val,
                "deviation": deviation,
                "correlation": corr,
                "signed_score": signed_score,
                "direction": "ai" if signed_score >= 0 else "human",
                "source": "global",
            }
        )

    lex_norm = float(
        np.mean(
            [
                normalized_features["lex_long_word_ratio"],
                normalized_features["lex_avg_word_len"],
            ]
        )
    )
    form_norm = float(
        np.mean(
            [
                normalized_features["form_connector_ratio"],
                normalized_features["form_first_person_ratio"],
                normalized_features["form_contraction_ratio"],
            ]
        )
    )
    burst_norm = float(
        np.mean(
            [
                normalized_features["burstiness"],
                normalized_features["syn_avg_sent_len"],
                normalized_features["syn_comma_per_sent"],
            ]
        )
    )

    global_scores = {
        "lexical_complexity": lex_norm,
        "formality": form_norm,
        "burstiness": burst_norm,
    }

    # 3) gradient-based saliency for both classes
    # Compute saliency for AI class (1)
    tokens_ai, tok_scores_ai, token_ids = get_token_importance(text, 1, max_length=max_length)
    tok_scores_ai = np.array(tok_scores_ai, dtype=float)
    if tok_scores_ai.max() > 0:
        tok_scores_ai = tok_scores_ai / tok_scores_ai.max()
    
    # Compute saliency for Human class (0)
    tokens_human, tok_scores_human, _ = get_token_importance(text, 0, max_length=max_length)
    tok_scores_human = np.array(tok_scores_human, dtype=float)
    if tok_scores_human.max() > 0:
        tok_scores_human = tok_scores_human / tok_scores_human.max()

    words, mapping = roberta_tokens_to_words_and_map(tokens_ai, token_ids)
    if len(words) == 0:
        return {
            "prediction": pred,
            "p_ai": p_ai,
            "global_scores": global_scores,
            "spans": [],
            "words": [],
        }

    # Map token saliency to words for both classes
    word_sal_ai = np.zeros(len(words), dtype=float)
    word_sal_human = np.zeros(len(words), dtype=float)
    
    for i, idxs in enumerate(mapping):
        idxs_ai = [k for k in idxs if k < len(tok_scores_ai)]
        idxs_human = [k for k in idxs if k < len(tok_scores_human)]
        if idxs_ai:
            word_sal_ai[i] = float(tok_scores_ai[idxs_ai].mean())
        if idxs_human:
            word_sal_human[i] = float(tok_scores_human[idxs_human].mean())
    
    # Normalize separately
    if word_sal_ai.max() > 0:
        word_sal_ai = word_sal_ai / word_sal_ai.max()
    if word_sal_human.max() > 0:
        word_sal_human = word_sal_human / word_sal_human.max()
    
    # Use the appropriate saliency based on prediction
    word_sal = word_sal_ai if pred == 1 else word_sal_human

    sent_ids = assign_sentence_ids(words)
    lex_loc, form_loc, burst_loc = get_word_level_feature_scores(words, sent_ids)

    def _nz_norm(x: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype=float)
        if x.max() > 0:
            x = x / x.max()
        return x

    lex_loc = _nz_norm(lex_loc)
    form_loc = _nz_norm(form_loc)
    burst_loc = _nz_norm(burst_loc)

    # Compute regression-based weights given p_ai and normalized features
    feature_groups = {
        "lexical": ["lex_long_word_ratio", "lex_avg_word_len"],
        "formality": ["form_first_person_ratio", "form_contraction_ratio"],
        "burstiness": ["burstiness", "syn_avg_sent_len", "syn_comma_per_sent"],
        "topic_uniformity": ["topic_uniformity"],
        "assertiveness": ["net_assertiveness"],
        "stopword": ["stopword_ratio"],
    }
    
    regression_weights = compute_regression_weights(
        normalized_features,
        p_ai,
        feature_groups
    )
    
    # Extract weights for each group
    lex_weight = regression_weights.get("lexical", 0.37)
    form_weight = regression_weights.get("formality", 0.085)
    burst_weight = regression_weights.get("burstiness", -0.16)
    topic_uniformity_weight = regression_weights.get("topic_uniformity", 0.05)
    assertiveness_weight = regression_weights.get("assertiveness", 0.15)
    stopword_weight = regression_weights.get("stopword", 0.191680)

    def _global_strength(score: float, weight: float) -> float:
        # For positive weights: high score → positive (AI), low score → negative (human)
        # For negative weights: high score → negative (human), low score → positive (AI)
        centered = (score - 0.5) * 2.0  # -1 .. 1
        return centered * weight

    lex_strength = _global_strength(global_scores["lexical_complexity"], lex_weight)
    form_strength = _global_strength(global_scores["formality"], form_weight)
    burst_strength = _global_strength(global_scores["burstiness"], burst_weight)

    topic_uniformity_strength = _global_strength(normalized_features.get("topic_uniformity", 0.5), topic_uniformity_weight)
    assertiveness_strength = _global_strength(normalized_features.get("net_assertiveness", 0.5), assertiveness_weight)
    stopword_strength = _global_strength(normalized_features.get("stopword_ratio", 0.5), stopword_weight)

    # Word-level contributions (form_loc already includes new patterns)
    n = len(words)
    lex_contrib_signed = lex_strength * lex_loc * word_sal
    form_contrib_signed = form_strength * form_loc * word_sal
    burst_contrib_signed = burst_strength * burst_loc * word_sal

    # Word-level stopword scores (1.0 if word is a stopword, 0.0 otherwise)
    lower_words = [w.lower() for w in words]
    stopword_loc = np.array([1.0 if w in STOP else 0.0 for w in lower_words], dtype=float)
    if stopword_loc.max() > 0:
        stopword_loc = stopword_loc / stopword_loc.max()
    stopword_contrib_signed = stopword_strength * stopword_loc * word_sal

    # Document-level contributions (distributed evenly across words for simplicity)
    topic_uniformity_contrib = (topic_uniformity_strength * np.ones(n) * word_sal) if n > 0 else np.zeros(n)
    assertiveness_contrib = (assertiveness_strength * np.ones(n) * word_sal) if n > 0 else np.zeros(n)

    lex_support_ai = np.maximum(0.0, lex_contrib_signed)
    lex_support_human = np.maximum(0.0, -lex_contrib_signed)
    form_support_ai = np.maximum(0.0, form_contrib_signed)
    form_support_human = np.maximum(0.0, -form_contrib_signed)
    burst_support_ai = np.maximum(0.0, burst_contrib_signed)
    burst_support_human = np.maximum(0.0, -burst_contrib_signed)
    
    # New feature supports
    topic_uniformity_support_ai = np.maximum(0.0, topic_uniformity_contrib)
    topic_uniformity_support_human = np.maximum(0.0, -topic_uniformity_contrib)
    assertiveness_support_ai = np.maximum(0.0, assertiveness_contrib)
    assertiveness_support_human = np.maximum(0.0, -assertiveness_contrib)
    stopword_support_ai = np.maximum(0.0, stopword_contrib_signed)
    stopword_support_human = np.maximum(0.0, -stopword_contrib_signed)

    total_ai_raw = (lex_support_ai + form_support_ai + burst_support_ai + 
                    topic_uniformity_support_ai + assertiveness_support_ai + stopword_support_ai)
    total_human_raw = (lex_support_human + form_support_human + burst_support_human +
                      topic_uniformity_support_human + assertiveness_support_human + stopword_support_human)
    total_ai = _nz_norm(total_ai_raw)
    total_human = _nz_norm(total_human_raw)

    support_norm = {1: total_ai, 0: total_human}
    support_raw = {1: total_ai_raw, 0: total_human_raw}
    lex_support_norm = {
        1: _nz_norm(lex_support_ai),
        0: _nz_norm(lex_support_human),
    }
    form_support_norm = {
        1: _nz_norm(form_support_ai),
        0: _nz_norm(form_support_human),
    }
    burst_support_norm = {
        1: _nz_norm(burst_support_ai),
        0: _nz_norm(burst_support_human),
    }

    total_signed = lex_contrib_signed + form_contrib_signed + burst_contrib_signed
    total_aligned = support_norm[pred]
    total_opposing = support_norm[1 - pred]
    total_aligned_raw = support_raw[pred]
    total_opposing_raw = support_raw[1 - pred]
    lex_contrib_aligned = lex_support_norm[pred]
    form_contrib_aligned = form_support_norm[pred]
    burst_contrib_aligned = burst_support_norm[pred]
    lex_contrib_opposing = lex_support_norm[1 - pred]
    form_contrib_opposing = form_support_norm[1 - pred]
    burst_contrib_opposing = burst_support_norm[1 - pred]

    spans_primary = select_top_spans(
        words,
        total_aligned,
        lex_contrib_aligned,
        form_contrib_aligned,
        burst_contrib_aligned,
        top_k=top_k,
        min_len=3,
        max_len=8,
        target_label="ai" if pred == 1 else "human",
    )
    spans_opposing = select_top_spans(
        words,
        total_opposing,
        lex_contrib_opposing,
        form_contrib_opposing,
        burst_contrib_opposing,
        top_k=top_k,
        min_len=3,
        max_len=8,
        target_label="human" if pred == 1 else "ai",
    )

    # --- Sentence-level aggregation ---
    sent_to_indices: Dict[int, List[int]] = {}
    for idx, sid in enumerate(sent_ids):
        sent_to_indices.setdefault(sid, []).append(idx)

    def _build_sentence_infos(raw_scores: np.ndarray) -> List[Dict[str, Any]]:
        infos: List[Dict[str, Any]] = []
        for sid, idxs in sent_to_indices.items():
            if not idxs:
                continue
            start = min(idxs)
            end = max(idxs) + 1
            sent_text = " ".join(words[start:end])
            raw_score = float(raw_scores[idxs].mean()) if len(idxs) > 0 else 0.0
            infos.append(
                {
                    "sentence_id": int(sid),
                    "start": int(start),
                    "end": int(end),
                    "text": sent_text,
                    "score_raw": raw_score,
                }
            )

        if infos:
            max_raw = max(s["score_raw"] for s in infos) or 1.0
            for s in infos:
                s["score"] = float(s["score_raw"] / max_raw) if max_raw > 0 else 0.0
        return infos

    sentence_infos = _build_sentence_infos(total_aligned_raw)
    sentence_infos_opposing = _build_sentence_infos(total_opposing_raw)

    # Aggregate impacts: positive signed_score means supports AI, negative means supports human
    # Lexical: high diversity (long words, high avg length) → positive → AI
    # Formality: depends on connectors/contractions mix
    # Burstiness: low burstiness (uniform sentences) → positive → AI, high burstiness → negative → human
    aggregate_impacts = [
        (
            "lexical_evidence",
            "Lexical Evidence",
            "High lexical diversity (long words, technical vocabulary) pushes toward AI classification.",
            float(lex_contrib_signed.sum()),
        ),
        (
            "formality_evidence",
            "Formality Evidence",
            "Formality cues such as connectors, contractions, and first-person voice.",
            float(form_contrib_signed.sum()),
        ),
        (
            "burstiness_evidence",
            "Burstiness Evidence",
            "Low burstiness (uniform sentence lengths) pushes toward AI; high burstiness (varied lengths) pushes toward human.",
            float(burst_contrib_signed.sum()),
        ),
        (
            "assertiveness_evidence",
            "Assertiveness Evidence",
            "Balance of assertive phrases (AI-like) vs hedging phrases (human-like).",
            float(assertiveness_contrib.sum()),
        ),
        (
            "stopword_evidence",
            "Stopword Ratio Evidence",
            "Ratio of common stopwords to total tokens. Higher ratio may indicate AI-generated text.",
            float(stopword_contrib_signed.sum()),
        ),
    ]

    for key, label, desc, value in aggregate_impacts:
        if value == 0:
            continue
        feature_impacts.insert(
            0,
            {
                "key": key,
                "label": label,
                "description": desc,
                "signed_score": value,
                "direction": "ai" if value >= 0 else "human",  # Positive = AI, Negative = Human
                "source": "local",
            },
        )

    feature_impacts.sort(key=lambda d: abs(d.get("signed_score", 0.0)), reverse=True)

    return {
        "prediction": pred,
        "p_ai": p_ai,
        "p_human": p_human,
        "global_scores": global_scores,
        "spans": spans_primary,
        "spans_opposing": spans_opposing,
        "words": words,

        # word-level importance (aligned with `words`)
        "word_importance": total_aligned.tolist(),
        "word_total_strength": total_aligned_raw.tolist(),
        "word_total_opposing": total_opposing_raw.tolist(),
        "word_support_ai": total_ai.tolist(),
        "word_support_human": total_human.tolist(),
        # gradient-based saliency (class-specific)
        "word_saliency_ai": word_sal_ai.tolist(),
        "word_saliency_human": word_sal_human.tolist(),
        "lex_support_ai": lex_support_norm[1].tolist(),
        "lex_support_human": lex_support_norm[0].tolist(),
        "form_support_ai": form_support_norm[1].tolist(),
        "form_support_human": form_support_norm[0].tolist(),
        "burst_support_ai": burst_support_norm[1].tolist(),
        "burst_support_human": burst_support_norm[0].tolist(),
        "word_contrib_aligned": total_aligned.tolist(),
        "word_contrib_signed": total_signed.tolist(),
        "lex_contrib": lex_contrib_aligned.tolist(),
        "form_contrib": form_contrib_aligned.tolist(),
        "burst_contrib": burst_contrib_aligned.tolist(),

        # sentence-level importance
        "sentences": sentence_infos,
        "sentences_opposing": sentence_infos_opposing,
        "feature_impacts": feature_impacts,
    }



# ============================================================
# END OF FILE
# ============================================================
