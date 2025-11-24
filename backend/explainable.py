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
}

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
) -> Tuple[List[str], np.ndarray]:
    """
    Returns tokens and importance scores for a single text.
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

    return tokens, token_importance


# ============================================================
# 4. TOKENS → WORDS + SENTENCE IDS
# ============================================================


def roberta_tokens_to_words_and_map(tokens: List[str]) -> Tuple[List[str], List[List[int]]]:
    """
    Turn RoBERTa tokens into words AND keep mapping.
    Returns:
      words   : list[str]
      mapping : list[list[int]]  (each word -> list of token indices)
    """
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
    else:
        lex_scores = lengths / max_len

    # formality: connectors & impersonal patterns get a bump
    form_scores = np.zeros(n, dtype=float)
    for i, w in enumerate(lower_words):
        if w in CONNECTOR_WORDS:
            form_scores[i] += 1.0
        if w in {"thus", "therefore", "consequently", "furthermore", "moreover"}:
            form_scores[i] += 1.0
        if w in {"we", "our"}:
            # first person plural used in academic style → small bump
            form_scores[i] += 0.3

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
    tokens, tok_scores = get_token_importance(text, target_class, max_length=max_length)
    tok_scores = np.array(tok_scores, dtype=float)
    if tok_scores.max() > 0:
        tok_scores = tok_scores / tok_scores.max()

    # 3) tokens → words
    words, mapping = roberta_tokens_to_words_and_map(tokens)
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

    This is handy for a Lovable / FastAPI UI where you just get one text.

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

    raw_feature_values = {
        "lex_long_word_ratio": float(lex["lex_long_word_ratio"]),
        "lex_avg_word_len": float(lex["lex_avg_word_len"]),
        "form_connector_ratio": float(form["form_connector_ratio"]),
        "form_first_person_ratio": float(form["form_first_person_ratio"]),
        "form_contraction_ratio": float(form["form_contraction_ratio"]),
        "syn_avg_sent_len": float(syn["syn_avg_sent_len"]),
        "syn_comma_per_sent": float(syn["syn_comma_per_sent"]),
        "burstiness": float(burst),
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

    # 3) gradient-based saliency
    tokens, tok_scores = get_token_importance(text, pred, max_length=max_length)
    tok_scores = np.array(tok_scores, dtype=float)
    if tok_scores.max() > 0:
        tok_scores = tok_scores / tok_scores.max()

    words, mapping = roberta_tokens_to_words_and_map(tokens)
    if len(words) == 0:
        return {
            "prediction": pred,
            "p_ai": p_ai,
            "global_scores": global_scores,
            "spans": [],
            "words": [],
        }

    word_sal = np.zeros(len(words), dtype=float)
    for i, idxs in enumerate(mapping):
        idxs = [k for k in idxs if k < len(tok_scores)]
        if not idxs:
            continue
        word_sal[i] = float(tok_scores[idxs].mean())
    if word_sal.max() > 0:
        word_sal = word_sal / word_sal.max()

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

    lex_weight = float(
        np.mean(
            [
                FEATURE_CORRELATIONS["lex_long_word_ratio"],
                FEATURE_CORRELATIONS["lex_avg_word_len"],
            ]
        )
    )
    form_weight = float(
        np.mean(
            [
                FEATURE_CORRELATIONS["form_first_person_ratio"],
                FEATURE_CORRELATIONS["form_contraction_ratio"],
            ]
        )
    )
    burst_weight = float(
        np.mean(
            [
                FEATURE_CORRELATIONS["burstiness"],
                FEATURE_CORRELATIONS["syn_avg_sent_len"],
                FEATURE_CORRELATIONS["syn_comma_per_sent"],
            ]
        )
    )

    def _global_strength(score: float, weight: float) -> float:
        centered = (score - 0.5) * 2.0  # -1 .. 1
        return centered * weight

    lex_strength = _global_strength(global_scores["lexical_complexity"], lex_weight)
    form_strength = _global_strength(global_scores["formality"], form_weight)
    burst_strength = _global_strength(global_scores["burstiness"], burst_weight)

    lex_contrib_signed = lex_strength * lex_loc * word_sal
    form_contrib_signed = form_strength * form_loc * word_sal
    burst_contrib_signed = burst_strength * burst_loc * word_sal

    target_sign = 1.0 if pred == 1 else -1.0
    lex_contrib_aligned = np.maximum(0.0, lex_contrib_signed * target_sign)
    form_contrib_aligned = np.maximum(0.0, form_contrib_signed * target_sign)
    burst_contrib_aligned = np.maximum(0.0, burst_contrib_signed * target_sign)

    total_signed = lex_contrib_signed + form_contrib_signed + burst_contrib_signed
    total_aligned = lex_contrib_aligned + form_contrib_aligned + burst_contrib_aligned

    if total_aligned.max() > 0:
        total_aligned = total_aligned / total_aligned.max()

    # --- Sentence-level aggregation ---
    sent_to_indices: Dict[int, List[int]] = {}
    for idx, sid in enumerate(sent_ids):
        sent_to_indices.setdefault(sid, []).append(idx)

    sentence_infos = []
    for sid, idxs in sent_to_indices.items():
        if not idxs:
            continue
        start = min(idxs)
        end = max(idxs) + 1
        sent_text = " ".join(words[start:end])
        sent_score = float(total_aligned[idxs].mean()) if len(idxs) > 0 else 0.0

        sentence_infos.append(
            {
                "sentence_id": int(sid),
                "start": int(start),  # word index start
                "end": int(end),  # word index end (exclusive)
                "text": sent_text,
                "score": sent_score,
            }
        )

    if sentence_infos:
        max_s = max(s["score"] for s in sentence_infos) or 1.0
        for s in sentence_infos:
            s["score"] = float(s["score"] / max_s)

    spans = select_top_spans(
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

    aggregate_impacts = [
        (
            "lexical_evidence",
            "Lexical Evidence",
            "Gradient-weighted lexical signals within highlighted spans.",
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
            "Sentence rhythm and burstiness compared to typical human variation.",
            float(burst_contrib_signed.sum()),
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
                "direction": "ai" if value >= 0 else "human",
                "source": "local",
            },
        )

    feature_impacts.sort(key=lambda d: abs(d.get("signed_score", 0.0)), reverse=True)

    return {
        "prediction": pred,
        "p_ai": p_ai,
        "p_human": p_human,
        "global_scores": global_scores,
        "spans": spans,
        "words": words,

        # word-level importance (aligned with `words`)
        "word_importance": total_aligned.tolist(),
        "word_contrib_aligned": total_aligned.tolist(),
        "word_contrib_signed": total_signed.tolist(),
        "lex_contrib": lex_contrib_aligned.tolist(),
        "form_contrib": form_contrib_aligned.tolist(),
        "burst_contrib": burst_contrib_aligned.tolist(),

        # sentence-level importance
        "sentences": sentence_infos,
        "feature_impacts": feature_impacts,
    }



# ============================================================
# END OF FILE
# ============================================================
