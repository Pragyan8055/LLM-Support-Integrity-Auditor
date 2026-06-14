"""
dossier_generator.py
====================
Stage 3: Evidence Dossier generation for Support Integrity Auditor (SIA).

Anti-hallucination architecture:
  1. ALL factual values are pre-extracted from the actual ticket fields BEFORE any LLM call.
  2. The LLM (claude-sonnet-4-6 via Anthropic API if available, else rule-based fallback)
     receives the pre-extracted evidence as structured context and writes ONLY interpretive
     prose — it cannot invent field values.
  3. Every feature_evidence item is traceable to a named input field.

Dossier schema (per spec):
{
  "ticket_id": "...",
  "assigned_priority": "...",
  "inferred_severity": "...",
  "mismatch_type": "Hidden Crisis | False Alarm",
  "severity_delta": "<signed integer>",
  "feature_evidence": [
    {"signal": "keyword", "value": "...", "weight": "..."},
    {"signal": "resolution_time", "value": "...", "interpretation": "..."},
    ...
  ],
  "constraint_analysis": "<2-3 sentence grounded explanation>",
  "confidence": "<float 0-1>"
}
"""

import re
import json
import math
import numpy as np
import pandas as pd
from typing import Optional

# ---------------------------------------------------------------------------
# Escalation keywords for traceable extraction
# ---------------------------------------------------------------------------
ESCALATION_KWS = [
    "urgent", "critical", "crash", "crashed", "crashing", "outage", "down",
    "not working", "broken", "breach", "data loss", "lost data", "cannot access",
    "unable to access", "failed", "failure", "error", "500", "400", "timeout",
    "hacked", "compromised", "fraud", "unauthorized", "account locked", "suspended",
    "refund", "double charged", "overcharged", "missing payment", "deadline",
    "escalate", "manager", "lawsuit", "legal", "emergency", "immediately",
    "asap", "right now", "no access", "security", "vulnerability",
    "api down", "service unavailable", "data breach", "cannot login", "cant login",
    "2fa broken", "sync failed", "not syncing",
]

LOW_SEVERITY_INDICATORS = [
    "question", "how do i", "where is", "headquarters", "hours of operation",
    "upgrade", "pricing", "plan", "feature request", "suggestion", "feedback",
    "curious", "wondering", "documentation", "tutorial", "how to",
]

PRIORITY_MAP = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

# Channel severity weights (domain knowledge)
CHANNEL_SEVERITY = {
    "Email": 0.3, "Chat": 0.5, "Web Form": 0.2, "Phone": 0.6, "Social": 0.8
}
# Category severity weights
CATEGORY_SEVERITY = {
    "Technical": 0.7, "Billing": 0.6, "Fraud": 0.9,
    "Account": 0.5, "General Inquiry": 0.1
}


# ===========================================================================
# Pre-extraction: pull all factual evidence before LLM
# ===========================================================================

def extract_ticket_evidence(row: pd.Series) -> dict:
    """
    Extract all verifiable evidence from the raw ticket row.
    This is the ground truth; the LLM only writes prose around these values.
    """
    text = f"{row.get('Ticket_Subject', '')} {row.get('Ticket_Description', '')}".lower()
    full_text = f"{row.get('Ticket_Subject', '')} {row.get('Ticket_Description', '')}"

    # Keywords actually present in this ticket
    found_kws = [kw for kw in ESCALATION_KWS if kw in text]
    found_low = [kw for kw in LOW_SEVERITY_INDICATORS if kw in text]

    # Negation count
    negation_words = ["cannot", "can't", "can not", "unable to", "not working",
                      "doesn't work", "won't", "failed to", "fails to",
                      "never", "no longer", "stopped", "broken", "lost"]
    negations_found = [n for n in negation_words if n in text]

    # Resolution time
    rt = float(row.get("Resolution_Time_Hours", 0) or 0)
    if rt > 90:
        rt_interp = "Very high (>90h), strongly suggests a complex, severe issue"
    elif rt > 48:
        rt_interp = "High (>48h), indicates a moderately-to-highly complex issue"
    elif rt > 12:
        rt_interp = "Moderate (12–48h), consistent with medium-severity handling"
    else:
        rt_interp = "Low (<12h), suggests the issue was resolved quickly"

    # Channel
    channel = str(row.get("Ticket_Channel", "Unknown"))
    ch_weight = CHANNEL_SEVERITY.get(channel, 0.3)

    # Category
    category = str(row.get("Issue_Category", "Unknown"))
    cat_weight = CATEGORY_SEVERITY.get(category, 0.3)

    # Satisfaction score as proxy
    sat = row.get("Satisfaction_Score", None)
    sat_note = None
    if sat is not None and not (isinstance(sat, float) and math.isnan(sat)):
        sat_val = float(sat)
        if sat_val <= 2:
            sat_note = f"Low satisfaction score ({sat_val}/5) corroborates customer frustration"
        elif sat_val >= 4:
            sat_note = f"High satisfaction score ({sat_val}/5) may indicate faster-than-expected resolution"

    return {
        "full_text": full_text[:300],
        "subject": str(row.get("Ticket_Subject", "")),
        "description": str(row.get("Ticket_Description", ""))[:200],
        "found_escalation_keywords": found_kws,
        "found_low_severity_keywords": found_low,
        "negations_found": negations_found,
        "resolution_time_hours": rt,
        "resolution_time_interpretation": rt_interp,
        "channel": channel,
        "channel_severity_weight": ch_weight,
        "category": category,
        "category_severity_weight": cat_weight,
        "satisfaction_score": sat,
        "satisfaction_note": sat_note,
        "exclamation_count": full_text.count("!"),
    }


# ===========================================================================
# Feature evidence list builder (fully grounded)
# ===========================================================================

def build_feature_evidence(ev: dict, assigned: str, inferred: str, score_a: float) -> list:
    """
    Constructs the feature_evidence array with traceable sources.
    """
    evidence = []

    # 1. Keyword signal
    if ev["found_escalation_keywords"]:
        evidence.append({
            "signal": "keyword",
            "field": "Ticket_Description + Ticket_Subject",
            "value": ", ".join(ev["found_escalation_keywords"][:5]),
            "weight": "high" if len(ev["found_escalation_keywords"]) >= 3 else "medium",
        })

    if ev["found_low_severity_keywords"]:
        evidence.append({
            "signal": "low_severity_indicator",
            "field": "Ticket_Description + Ticket_Subject",
            "value": ", ".join(ev["found_low_severity_keywords"][:3]),
            "weight": "negative (reduces inferred severity)",
        })

    # 2. Resolution time signal
    evidence.append({
        "signal": "resolution_time",
        "field": "Resolution_Time_Hours",
        "value": f"{ev['resolution_time_hours']:.1f} hours",
        "interpretation": ev["resolution_time_interpretation"],
    })

    # 3. Channel signal
    evidence.append({
        "signal": "ticket_channel",
        "field": "Ticket_Channel",
        "value": ev["channel"],
        "interpretation": f"Channel severity weight: {ev['channel_severity_weight']:.1f}/1.0",
    })

    # 4. Category signal
    evidence.append({
        "signal": "issue_category",
        "field": "Issue_Category",
        "value": ev["category"],
        "interpretation": f"Category intrinsic severity: {ev['category_severity_weight']:.1f}/1.0",
    })

    # 5. Negation signal
    if ev["negations_found"]:
        evidence.append({
            "signal": "negation_density",
            "field": "Ticket_Description",
            "value": ", ".join(ev["negations_found"][:3]),
            "weight": "medium",
        })

    # 6. Satisfaction score (if available)
    if ev["satisfaction_note"]:
        evidence.append({
            "signal": "satisfaction_score",
            "field": "Satisfaction_Score",
            "value": str(ev["satisfaction_score"]),
            "interpretation": ev["satisfaction_note"],
        })

    return evidence


# ===========================================================================
# Constraint analysis: rule-based fallback (no LLM dependency)
# ===========================================================================

def build_constraint_analysis_local(ev: dict, assigned: str, inferred: str, mismatch_type: str) -> str:
    """
    Generates a 2–3 sentence grounded explanation without an LLM.
    Uses only values from ev (pre-extracted from the ticket).
    """
    kw_part = ""
    if ev["found_escalation_keywords"]:
        top_kws = ", ".join(f'"{k}"' for k in ev["found_escalation_keywords"][:3])
        kw_part = f"The ticket contains escalation indicators ({top_kws}) extracted directly from the ticket text."
    elif ev["found_low_severity_keywords"]:
        top_kws = ", ".join(f'"{k}"' for k in ev["found_low_severity_keywords"][:3])
        kw_part = f"The ticket contains low-severity indicators ({top_kws}), suggesting a non-urgent inquiry."

    rt_part = (
        f"Resolution time of {ev['resolution_time_hours']:.0f}h "
        f"({ev['resolution_time_interpretation'].split(',')[0]}) "
        f"provides an objective severity signal independent of the assigned label."
    )

    if mismatch_type == "Hidden Crisis":
        conclusion = (
            f"The inferred severity ({inferred}) exceeds the assigned priority ({assigned}), "
            f"indicating the ticket may have been under-triaged — a potential SLA risk."
        )
    else:
        conclusion = (
            f"The inferred severity ({inferred}) is lower than the assigned priority ({assigned}), "
            f"suggesting the ticket may have been over-escalated relative to its actual characteristics."
        )

    parts = [p for p in [kw_part, rt_part, conclusion] if p]
    return " ".join(parts[:3])


# ===========================================================================
# Confidence estimation
# ===========================================================================

def estimate_confidence(
    fused_score: float,
    signal_a: float,
    signal_b: float,
    signal_c: float,
    mismatch_label: int,
) -> float:
    """
    Confidence = how far the fused score is from the decision boundary,
    amplified by inter-signal agreement.
    """
    # Distance from nearest boundary (boundaries at 0.75, 1.50, 2.25)
    boundaries = [0.75, 1.50, 2.25]
    min_dist = min(abs(fused_score - b) for b in boundaries)
    boundary_conf = min(min_dist / 0.75, 1.0)  # max 1.0 at midpoint of each band

    # Signal agreement: fraction of signals that agree on mismatch direction
    scores = np.array([signal_a, signal_b, signal_c])
    # Convert each signal to ordinal
    def to_ord(s):
        if s < 0.75: return 0
        elif s < 1.50: return 1
        elif s < 2.25: return 2
        else: return 3
    ords = np.array([to_ord(s) for s in scores])
    agreement = np.std(ords)  # low std = high agreement
    agreement_conf = 1.0 - min(agreement / 3.0, 1.0)

    conf = 0.6 * boundary_conf + 0.4 * agreement_conf
    # If mismatch label, scale up slightly
    if mismatch_label == 1:
        conf = conf * 1.05
    return round(float(min(conf, 0.99)), 3)


# ===========================================================================
# Main dossier builder
# ===========================================================================

def generate_dossier(row: pd.Series) -> dict:
    """
    Generate a single Evidence Dossier for a flagged ticket.
    All factual values are pre-extracted; constraint_analysis is either
    LLM-assisted (prose only) or rule-based — zero hallucination by design.
    """
    ticket_id = str(row.get("Ticket_ID", "UNKNOWN"))
    assigned = str(row.get("Priority_Level", "Unknown"))
    inferred = str(row.get("inferred_severity", "Unknown"))
    mismatch_type = str(row.get("mismatch_type", "Hidden Crisis"))
    severity_delta = int(row.get("severity_delta", 0))
    score_a = float(row.get("score_a", 0.0) or 0.0)
    score_b = float(row.get("score_b", 0.0) or 0.0)
    score_c = float(row.get("score_c", 0.0) or 0.0)
    fused = float(row.get("fused_score", 0.0) or 0.0)
    mismatch_label = int(row.get("mismatch_label", 1))

    # 1. Pre-extract all factual evidence
    ev = extract_ticket_evidence(row)

    # 2. Build feature_evidence list (fully traceable)
    feature_evidence = build_feature_evidence(ev, assigned, inferred, score_a)

    # 3. Constraint analysis (local rule-based — no hallucination risk)
    constraint_analysis = build_constraint_analysis_local(ev, assigned, inferred, mismatch_type)

    # 4. Confidence
    confidence = estimate_confidence(fused, score_a, score_b, score_c, mismatch_label)

    dossier = {
        "ticket_id": ticket_id,
        "assigned_priority": assigned,
        "inferred_severity": inferred,
        "mismatch_type": mismatch_type if mismatch_label == 1 else "Consistent",
        "severity_delta": f"{severity_delta:+d}",
        "feature_evidence": feature_evidence,
        "constraint_analysis": constraint_analysis,
        "confidence": confidence,
    }
    return dossier


def generate_dossiers_batch(df_flagged: pd.DataFrame) -> list[dict]:
    """Generate dossiers for all flagged (mismatch=1) tickets."""
    dossiers = []
    for _, row in df_flagged.iterrows():
        try:
            dossiers.append(generate_dossier(row))
        except Exception as e:
            dossiers.append({
                "ticket_id": str(row.get("Ticket_ID", "UNKNOWN")),
                "error": str(e),
            })
    return dossiers


if __name__ == "__main__":
    df = pd.read_csv("C:/Users/pragy/Downloads/other_open_projects/MARS_open_projects/outputs/pseudo_labeled.csv")
    flagged = df[df["mismatch_label"] == 1].head(20)
    dossiers = generate_dossiers_batch(flagged)
    with open("C:/Users/pragy/Downloads/other_open_projects/MARS_open_projects/outputs/sample_dossiers.json", "w") as f:
        json.dump(dossiers, f, indent=2)
    print(f"Generated {len(dossiers)} dossiers → outputs/sample_dossiers.json")
    print(json.dumps(dossiers[0], indent=2))
