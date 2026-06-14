"""
pseudo_labeler.py
=================
Stage 1: Self-supervised pseudo-label generation for the Support Integrity Auditor (SIA).

Three independent signals are fused to infer the "true" severity of a ticket,
independent of its human-assigned Ticket Priority:

  Signal A — Rule-based NLP features (keyword density, negation, escalation phrases,
              sentiment via VADER)
  Signal B — Resolution-time regression (XGBoost regressor trained on NLP features;
              predicted resolution time as a severity proxy)
  Signal C — Embedding-based clustering (sentence-transformers MiniLM embeddings,
              K-Means; cluster urgency scores derived from centroid text analysis)

Fusion: weighted average [0.45 A + 0.30 B + 0.25 C], mapped to Low/Medium/High/Critical.
Binary mismatch label = 1 if inferred severity disagrees with assigned Priority_Level.

Ablation logging is written to outputs/ablation.json.
"""

import os
import json
import re
import warnings
import numpy as np
import pandas as pd
from typing import Optional

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Priority / severity ordinal mapping
# ---------------------------------------------------------------------------
PRIORITY_MAP = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
INV_PRIORITY_MAP = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}


# ===========================================================================
# SIGNAL A: Rule-based NLP features
# ===========================================================================

ESCALATION_KEYWORDS = [
    "urgent", "critical", "crash", "crashed", "crashing", "outage", "down",
    "not working", "broken", "breach", "data loss", "lost data", "cannot access",
    "unable to access", "failed", "failure", "error", "500", "400", "timeout",
    "hacked", "compromised", "fraud", "unauthorized", "account locked", "suspended",
    "refund", "double charged", "overcharged", "missing payment", "deadline",
    "sla", "escalate", "manager", "lawsuit", "legal", "emergency", "immediately",
    "asap", "right now", "no access", "security", "vulnerability", "exploit",
    "api down", "service unavailable", "data breach", "cannot login", "cant login",
    "2fa broken", "sync failed", "not syncing",
]

NEGATION_TRIGGERS = [
    "cannot", "can't", "can not", "unable to", "not working", "doesn't work",
    "won't", "will not", "failed to", "fails to", "never", "no longer",
    "stopped", "broken", "lost",
]

LOW_SEVERITY_INDICATORS = [
    "question", "how do i", "where is", "headquarters", "hours of operation",
    "upgrade", "pricing", "plan", "feature request", "suggestion", "feedback",
    "curious", "wondering", "documentation", "tutorial", "how to",
]


def _vader_sentiment(text: str) -> float:
    """Return compound VADER sentiment score (-1 to 1); lower = more negative = more urgent."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
        return _vader.polarity_scores(text)["compound"]
    except Exception:
        return 0.0


def _rule_score(subject: str, description: str) -> float:
    """
    Returns a severity score in [0, 3] based on rule-based NLP features.
    """
    text = f"{subject} {description}".lower()

    # 1. Escalation keyword density
    kw_hits = sum(1 for kw in ESCALATION_KEYWORDS if kw in text)
    kw_score = min(kw_hits / 3.0, 1.0)  # normalise to [0,1]

    # 2. Negation count
    neg_hits = sum(1 for n in NEGATION_TRIGGERS if n in text)
    neg_score = min(neg_hits / 2.0, 1.0)

    # 3. Low-severity indicator penalty
    low_hits = sum(1 for l in LOW_SEVERITY_INDICATORS if l in text)
    low_penalty = min(low_hits * 0.3, 0.6)

    # 4. Sentiment (negative = urgent)
    sentiment = _vader_sentiment(text)
    sentiment_score = max(0.0, (-sentiment + 1) / 2)  # map [-1,1] -> [1,0] -> [0,1]

    # 5. Exclamation / all-caps signals
    exclamation_score = min(text.count("!") * 0.15, 0.3)

    raw = (0.40 * kw_score + 0.25 * neg_score + 0.20 * sentiment_score
           + 0.15 * exclamation_score) - low_penalty
    raw = max(0.0, min(1.0, raw))
    return raw * 3.0  # scale to [0, 3]


def compute_signal_a(df: pd.DataFrame) -> np.ndarray:
    """Signal A: rule-based NLP scores (0–3) for each ticket."""
    scores = []
    for _, row in df.iterrows():
        scores.append(_rule_score(
            str(row.get("Ticket_Subject", "")),
            str(row.get("Ticket_Description", ""))
        ))
    return np.array(scores)


# ===========================================================================
# SIGNAL B: Resolution-time regression
# ===========================================================================

def compute_signal_b(df: pd.DataFrame, signal_a_scores: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Signal B: XGBoost regression on NLP features predicts resolution time.
    Predicted resolution time is then inverted to a severity score (0–3).
    Higher predicted resolution time → higher inferred severity.
    """
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_predict

    feat_df = pd.DataFrame()

    # Text-derived features
    texts = (df["Ticket_Subject"].fillna("") + " " + df["Ticket_Description"].fillna("")).str.lower()
    feat_df["text_len"] = texts.str.len()
    feat_df["word_count"] = texts.str.split().str.len()
    feat_df["exclamation"] = texts.str.count("!")
    feat_df["kw_hits"] = texts.apply(
        lambda t: sum(1 for kw in ESCALATION_KEYWORDS if kw in t)
    )
    feat_df["neg_hits"] = texts.apply(
        lambda t: sum(1 for n in NEGATION_TRIGGERS if n in t)
    )
    feat_df["low_hits"] = texts.apply(
        lambda t: sum(1 for l in LOW_SEVERITY_INDICATORS if l in t)
    )

    # Structured features
    le_cat = LabelEncoder()
    le_ch = LabelEncoder()
    feat_df["issue_category"] = le_cat.fit_transform(df["Issue_Category"].fillna("Unknown"))
    feat_df["channel"] = le_ch.fit_transform(df["Ticket_Channel"].fillna("Unknown"))

    if signal_a_scores is not None:
        feat_df["signal_a"] = signal_a_scores

    X = feat_df.values
    y = df["Resolution_Time_Hours"].fillna(df["Resolution_Time_Hours"].median()).values

    model = XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, verbosity=0
    )

    # CV predictions to avoid overfitting leakage
    y_pred = cross_val_predict(model, X, y, cv=5)

    # Normalise to [0, 3]: higher time → higher severity
    y_min, y_max = y_pred.min(), y_pred.max()
    norm = (y_pred - y_min) / (y_max - y_min + 1e-9) * 3.0
    return norm.clip(0, 3)


# ===========================================================================
# SIGNAL C: Embedding-based clustering
# ===========================================================================

def compute_signal_c(df: pd.DataFrame, n_clusters: int = 8) -> np.ndarray:
    """
    Signal C: Sentence-transformer embeddings + K-Means clustering.
    Each cluster gets an urgency score derived from keyword density of member tickets.
    Returns per-ticket severity scores (0–3).
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans, MiniBatchKMeans

    texts = (
        df["Ticket_Subject"].fillna("") + ". " + df["Ticket_Description"].fillna("")
    ).tolist()

    # Use a tiny, fast model suited for CPU
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Encode in batches to avoid memory spike
    batch_size = 256
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings_list.append(emb)
    embeddings = np.vstack(embeddings_list)

    # Cluster
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(embeddings)

    # Score each cluster by mean keyword hit rate of its members
    kw_hits = np.array([
        sum(1 for kw in ESCALATION_KEYWORDS if kw in t.lower())
        for t in texts
    ])
    cluster_urgency = {}
    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_urgency[c] = kw_hits[mask].mean() if mask.sum() > 0 else 0.0

    # Map cluster urgency to [0, 3]
    max_urg = max(cluster_urgency.values()) + 1e-9
    scores = np.array([cluster_urgency[c] / max_urg * 3.0 for c in cluster_labels])
    return scores.clip(0, 3)


# ===========================================================================
# FUSION + MISMATCH LABEL
# ===========================================================================

FUSION_WEIGHTS = {"signal_a": 0.45, "signal_b": 0.30, "signal_c": 0.25}


def _score_to_priority(score: float) -> str:
    """Map a continuous [0, 3] score to a priority label."""
    if score < 0.75:
        return "Low"
    elif score < 1.50:
        return "Medium"
    elif score < 2.25:
        return "High"
    else:
        return "Critical"


def fuse_signals(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    signal_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted average fusion → inferred severity (continuous + label).
    Returns (continuous_scores, label_array).
    """
    fused = (
        FUSION_WEIGHTS["signal_a"] * signal_a
        + FUSION_WEIGHTS["signal_b"] * signal_b
        + FUSION_WEIGHTS["signal_c"] * signal_c
    )
    labels = np.array([_score_to_priority(s) for s in fused])
    return fused, labels


def compute_mismatch_label(
    assigned: pd.Series, inferred: np.ndarray
) -> np.ndarray:
    """
    Binary mismatch: 1 if |ordinal(assigned) - ordinal(inferred)| >= 1.
    """
    assigned_ord = np.array([PRIORITY_MAP.get(p, 1) for p in assigned])
    inferred_ord = np.array([PRIORITY_MAP.get(p, 1) for p in inferred])
    return (np.abs(assigned_ord - inferred_ord) >= 1).astype(int)


# ===========================================================================
# PAIRWISE SIGNAL AGREEMENT (for ablation)
# ===========================================================================

def pairwise_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of tickets where two signals give the same ordinal bucket."""
    a_lbl = np.array([_score_to_priority(s) for s in a])
    b_lbl = np.array([_score_to_priority(s) for s in b])
    return float((a_lbl == b_lbl).mean())


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

def generate_pseudo_labels(
    df: pd.DataFrame,
    output_dir: str = "outputs",
) -> pd.DataFrame:
    """
    Full pseudo-label pipeline. Returns df with added columns:
      score_a, score_b, score_c, fused_score, inferred_severity, mismatch_label,
      severity_delta, mismatch_type.
    Also writes ablation.json to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("[SIA] Computing Signal A (rule-based NLP)...")
    sa = compute_signal_a(df)

    print("[SIA] Computing Signal B (resolution-time regression)...")
    sb = compute_signal_b(df, signal_a_scores=sa)

    print("[SIA] Computing Signal C (embedding clustering)...")
    sc = compute_signal_c(df)

    print("[SIA] Fusing signals...")
    fused, inferred_labels = fuse_signals(sa, sb, sc)
    mismatch = compute_mismatch_label(df["Priority_Level"], inferred_labels)

    # Severity delta (signed: positive = under-triaged = Hidden Crisis)
    assigned_ord = np.array([PRIORITY_MAP.get(p, 1) for p in df["Priority_Level"]])
    inferred_ord = np.array([PRIORITY_MAP.get(p, 1) for p in inferred_labels])
    delta = inferred_ord - assigned_ord  # >0 = true severity HIGHER than assigned

    mismatch_type = np.where(delta > 0, "Hidden Crisis", np.where(delta < 0, "False Alarm", "Consistent"))

    df = df.copy()
    df["score_a"] = sa
    df["score_b"] = sb
    df["score_c"] = sc
    df["fused_score"] = fused
    df["inferred_severity"] = inferred_labels
    df["mismatch_label"] = mismatch
    df["severity_delta"] = delta
    df["mismatch_type"] = mismatch_type

    # Ablation: per-signal standalone performance + pairwise agreement
    ablation = {
        "fusion_weights": FUSION_WEIGHTS,
        "mismatch_rate_overall": float(mismatch.mean()),
        "pairwise_agreement": {
            "A_vs_B": pairwise_agreement(sa, sb),
            "A_vs_C": pairwise_agreement(sa, sc),
            "B_vs_C": pairwise_agreement(sb, sc),
        },
        "individual_signal_mismatch_rate": {
            "signal_a_only": float(compute_mismatch_label(
                df["Priority_Level"], np.array([_score_to_priority(s) for s in sa])
            ).mean()),
            "signal_b_only": float(compute_mismatch_label(
                df["Priority_Level"], np.array([_score_to_priority(s) for s in sb])
            ).mean()),
            "signal_c_only": float(compute_mismatch_label(
                df["Priority_Level"], np.array([_score_to_priority(s) for s in sc])
            ).mean()),
        },
        "class_distribution": {
            "mismatch_count": int(mismatch.sum()),
            "consistent_count": int((mismatch == 0).sum()),
            "mismatch_type_counts": {
                "Hidden Crisis": int((mismatch_type == "Hidden Crisis").sum()),
                "False Alarm": int((mismatch_type == "False Alarm").sum()),
                "Consistent": int((mismatch_type == "Consistent").sum()),
            }
        }
    }

    with open(os.path.join(output_dir, "ablation.json"), "w") as f:
        json.dump(ablation, f, indent=2)

    print(f"[SIA] Pseudo-labeling complete.")
    print(f"      Mismatch rate: {mismatch.mean():.2%}")
    print(f"      Hidden Crisis: {(mismatch_type == 'Hidden Crisis').sum()}")
    print(f"      False Alarm:   {(mismatch_type == 'False Alarm').sum()}")
    print(f"      Ablation saved to {output_dir}/ablation.json")

    return df

if __name__ == "__main__":
    df = pd.read_csv("C:/Users/pragy/Downloads/other_open_projects/MARS_open_projects/customer_support_tickets.csv")
    result = generate_pseudo_labels(df, output_dir="C:/Users/pragy/Downloads/other_open_projects/MARS_open_projects/outputs")
    result.to_csv("C:/Users/pragy/Downloads/other_open_projects/MARS_open_projects/outputs/pseudo_labeled.csv", index=False)
    print("Saved outputs/pseudo_labeled.csv")
