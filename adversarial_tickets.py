"""
adversarial_tickets.py
======================
10 held-out adversarial tickets designed to expose failures in keyword-based triage.

Pattern types:
  A) High true severity, ZERO escalation keywords (fool keyword systems into Low)
  B) Zero true severity, MANY escalation keywords (fool keyword systems into Critical)
  C) Negated severity phrases (e.g., "NOT crashed", "no errors")
  D) Context-dependent severity (product tier, domain)

Ground truth (GT) is the label a semantic system should assign.
"""

import pandas as pd

ADVERSARIAL_TICKETS = [
    # ── Type A: Hidden Crisis (no keywords, true severity = Critical) ──────────

    {
        "Ticket_ID": "ADV-001",
        "Ticket_Subject": "Quick question about my records",
        "Ticket_Description": (
            "Hi, I wanted to check something. I tried to open our shared workspace "
            "yesterday and today, and it appears that all files from the past three "
            "months are no longer visible. My colleagues report the same situation. "
            "We have a client delivery tomorrow morning. Could someone look into this?"
        ),
        "Priority_Level": "Low",        # human-assigned (under-triaged)
        "Issue_Category": "Technical",
        "Ticket_Channel": "Email",
        "Resolution_Time_Hours": 96,
        "Satisfaction_Score": 1,
        "GT_mismatch": 1,
        "GT_type": "Hidden Crisis",
        "GT_inferred": "Critical",
        "adversarial_pattern": "No escalation keywords; describes complete data loss"
    },
    {
        "Ticket_ID": "ADV-002",
        "Ticket_Subject": "Billing question",
        "Ticket_Description": (
            "I noticed a discrepancy in my invoice from last cycle. The amount charged "
            "was approximately three times what our contract specifies. We are a team "
            "of 200 users on the Enterprise plan. I would appreciate clarification "
            "before our procurement team initiates a dispute process."
        ),
        "Priority_Level": "Medium",     # under-triaged
        "Issue_Category": "Billing",
        "Ticket_Channel": "Email",
        "Resolution_Time_Hours": 72,
        "Satisfaction_Score": 2,
        "GT_mismatch": 1,
        "GT_type": "Hidden Crisis",
        "GT_inferred": "High",
        "adversarial_pattern": "No escalation words; describes 3x overcharge on Enterprise account"
    },
    {
        "Ticket_ID": "ADV-003",
        "Ticket_Subject": "Login assistance needed",
        "Ticket_Description": (
            "Our primary system administrator account appears to have been modified "
            "without authorisation from our side. The associated email was changed "
            "and we have been unable to regain entry for 18 hours. This account "
            "controls access for our entire organisation of 500 staff."
        ),
        "Priority_Level": "Low",        # massively under-triaged
        "Issue_Category": "Account",
        "Ticket_Channel": "Web Form",
        "Resolution_Time_Hours": 110,
        "Satisfaction_Score": 1,
        "GT_mismatch": 1,
        "GT_type": "Hidden Crisis",
        "GT_inferred": "Critical",
        "adversarial_pattern": "Calm phrasing concealing account takeover for 500-user org"
    },
    {
        "Ticket_ID": "ADV-004",
        "Ticket_Subject": "API response times",
        "Ticket_Description": (
            "The average response latency on our production API endpoints has increased "
            "from 120ms to approximately 8,200ms over the past six hours. Our "
            "end-users are experiencing timeouts. We process roughly 40,000 "
            "transactions per hour through this integration."
        ),
        "Priority_Level": "Medium",     # under-triaged
        "Issue_Category": "Technical",
        "Ticket_Channel": "Chat",
        "Resolution_Time_Hours": 88,
        "Satisfaction_Score": 1,
        "GT_mismatch": 1,
        "GT_type": "Hidden Crisis",
        "GT_inferred": "Critical",
        "adversarial_pattern": "No alarm words; 68x latency degradation on high-volume production system"
    },
    {
        "Ticket_ID": "ADV-005",
        "Ticket_Subject": "Routine account inquiry",
        "Ticket_Description": (
            "During our quarterly security audit we identified that several former "
            "employees still retain active login credentials to our environment, "
            "including one individual who departed after a disciplinary matter. "
            "We would like guidance on how to proceed with access revocation at scale."
        ),
        "Priority_Level": "Low",        # under-triaged security issue
        "Issue_Category": "Account",
        "Ticket_Channel": "Email",
        "Resolution_Time_Hours": 60,
        "Satisfaction_Score": 3,
        "GT_mismatch": 1,
        "GT_type": "Hidden Crisis",
        "GT_inferred": "High",
        "adversarial_pattern": "Calm audit framing conceals active insider-threat exposure"
    },

    # ── Type B: False Alarm (many keywords, true severity = Low) ─────────────

    {
        "Ticket_ID": "ADV-006",
        "Ticket_Subject": "URGENT!!! CRITICAL ERROR OUTAGE EMERGENCY!!!",
        "Ticket_Description": (
            "EMERGENCY!!! The button in the top-right corner of the dashboard is "
            "displaying in blue instead of the correct brand colour (hex #2A3F8D). "
            "This is CRITICAL for our brand guidelines. Please fix ASAP!!!"
        ),
        "Priority_Level": "Critical",   # over-triaged
        "Issue_Category": "General Inquiry",
        "Ticket_Channel": "Chat",
        "Resolution_Time_Hours": 2,
        "Satisfaction_Score": 5,
        "GT_mismatch": 1,
        "GT_type": "False Alarm",
        "GT_inferred": "Low",
        "adversarial_pattern": "Maximum keyword density; describes cosmetic colour complaint"
    },
    {
        "Ticket_ID": "ADV-007",
        "Ticket_Subject": "SYSTEM DOWN BREACH FRAUD HACK",
        "Ticket_Description": (
            "Hi! Our marketing team is planning a campaign and we were wondering if "
            "you offer bulk pricing for additional user seats. We are currently on "
            "the Pro plan and are thinking about upgrading. No rush on this at all, "
            "just checking our options for next quarter!"
        ),
        "Priority_Level": "High",       # over-triaged
        "Issue_Category": "General Inquiry",
        "Ticket_Channel": "Email",
        "Resolution_Time_Hours": 4,
        "Satisfaction_Score": 5,
        "GT_mismatch": 1,
        "GT_type": "False Alarm",
        "GT_inferred": "Low",
        "adversarial_pattern": "Misleading subject with escalation words; body is a sales inquiry"
    },
    {
        "Ticket_ID": "ADV-008",
        "Ticket_Subject": "Critical failure unauthorized access data loss outage",
        "Ticket_Description": (
            "I was wondering if you have any tutorial videos on how to use the "
            "reporting module? I'm new to the platform and would love some guidance. "
            "The documentation is great but I am a visual learner. Thank you!"
        ),
        "Priority_Level": "High",       # over-triaged
        "Issue_Category": "General Inquiry",
        "Ticket_Channel": "Chat",
        "Resolution_Time_Hours": 3,
        "Satisfaction_Score": 5,
        "GT_mismatch": 1,
        "GT_type": "False Alarm",
        "GT_inferred": "Low",
        "adversarial_pattern": "All escalation keywords in subject; body is a tutorial request"
    },

    # ── Type C: Negation adversarial ─────────────────────────────────────────

    {
        "Ticket_ID": "ADV-009",
        "Ticket_Subject": "No crash, no error, no problem — just a feature request",
        "Ticket_Description": (
            "Everything is working perfectly. No outage, no failure, no data loss. "
            "I simply wanted to suggest that it would be nice to have a dark mode "
            "option in the settings panel. Not urgent, just a low-priority idea "
            "for your product roadmap."
        ),
        "Priority_Level": "High",       # over-triaged
        "Issue_Category": "General Inquiry",
        "Ticket_Channel": "Email",
        "Resolution_Time_Hours": 1,
        "Satisfaction_Score": 5,
        "GT_mismatch": 1,
        "GT_type": "False Alarm",
        "GT_inferred": "Low",
        "adversarial_pattern": "Negated escalation words throughout; pure feature suggestion"
    },
    {
        "Ticket_ID": "ADV-010",
        "Ticket_Subject": "Follow-up on resolved issue",
        "Ticket_Description": (
            "I'm writing to confirm that the payment processing issue we reported "
            "last week has been fully resolved from our side. Our transactions are "
            "processing normally again. Just a courtesy note — no action needed. "
            "The team was very helpful. Thank you for the quick resolution."
        ),
        "Priority_Level": "Critical",   # over-triaged (past-tense resolved issue)
        "Issue_Category": "Billing",
        "Ticket_Channel": "Email",
        "Resolution_Time_Hours": 1,
        "Satisfaction_Score": 5,
        "GT_mismatch": 1,
        "GT_type": "False Alarm",
        "GT_inferred": "Low",
        "adversarial_pattern": "Resolved past issue marked Critical; semantic context = thank-you note"
    },
]


def get_adversarial_df() -> pd.DataFrame:
    return pd.DataFrame(ADVERSARIAL_TICKETS)


def run_adversarial_eval(predict_fn) -> dict:
    """
    Run the adversarial set through the full predict pipeline.
    Returns accuracy score and per-ticket results.
    """
    df = get_adversarial_df()
    result_df = predict_fn(df)

    correct = 0
    results = []
    for i, row in result_df.iterrows():
        orig = df.loc[i]
        pred_mismatch = int(row.get("predicted_mismatch", row.get("mismatch_label", 0)))
        gt_mismatch = int(orig["GT_mismatch"])
        is_correct = pred_mismatch == gt_mismatch
        if is_correct:
            correct += 1
        results.append({
            "ticket_id": orig["Ticket_ID"],
            "pattern": orig["adversarial_pattern"],
            "gt_mismatch": gt_mismatch,
            "predicted_mismatch": pred_mismatch,
            "correct": is_correct,
            "gt_type": orig["GT_type"],
        })

    score = correct / len(df)
    return {
        "adversarial_score": score,
        "correct": correct,
        "total": len(df),
        "bonus_earned": score >= 0.7,
        "per_ticket": results,
    }


if __name__ == "__main__":
    df = get_adversarial_df()
    print(f"Adversarial test set: {len(df)} tickets")
    print(df[["Ticket_ID", "GT_type", "GT_inferred", "adversarial_pattern"]].to_string())
