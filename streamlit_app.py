"""
streamlit_app.py
================
Support Integrity Auditor (SIA) — Streamlit Web Application

Views:
  1. Single Ticket Audit   — form input → mismatch judgment + Evidence Dossier
  2. Batch CSV Upload      — upload CSV → predictions + download + summary stats
  3. Priority Mismatch Dashboard — distribution charts, top signals
  4. Severity Delta Heatmap — Ticket Type × Channel grid

Run:
  cd sia_project && streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Support Integrity Auditor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.main-title { font-size:2rem; font-weight:700; color:#1a1a2e; margin-bottom:0; }
.subtitle   { font-size:1rem; color:#555; margin-top:0; margin-bottom:1.5rem; }
.metric-card {
    background:#f8f9fa; border-radius:8px; padding:16px;
    border-left:4px solid #4361ee; margin-bottom:12px;
}
.crisis-badge {
    background:#ff4d4d; color:white; padding:4px 10px;
    border-radius:12px; font-size:0.85rem; font-weight:600;
}
.alarm-badge {
    background:#ff9500; color:white; padding:4px 10px;
    border-radius:12px; font-size:0.85rem; font-weight:600;
}
.consistent-badge {
    background:#28a745; color:white; padding:4px 10px;
    border-radius:12px; font-size:0.85rem; font-weight:600;
}
.evidence-block {
    background:#f0f4ff; border-radius:6px; padding:12px;
    border-left:3px solid #4361ee; margin:6px 0; font-size:0.9rem;
}
.dossier-section { margin-top:1rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/64/search-property.png", width=50)
    st.markdown("### 🔍 Support Integrity Auditor")
    st.markdown("*Self-supervised priority mismatch detection*")
    st.divider()
    page = st.radio(
        "Navigation",
        ["🎫 Single Ticket Audit", "📂 Batch CSV Upload",
         "📊 Mismatch Dashboard", "🌡️ Severity Heatmap"],
    )
    st.divider()
    st.markdown("**Model:** DeBERTa-v3-small + LoRA")
    st.markdown("**Pipeline:** 3-signal pseudo-labeling → fine-tuned classifier → Evidence Dossier")

    # Load metrics if available
    metrics_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        st.divider()
        st.markdown("**Validation Metrics**")
        acc = metrics.get("binary_accuracy", 0)
        f1  = metrics.get("macro_f1", 0)
        rc  = metrics.get("recall_consistent", 0)
        rm  = metrics.get("recall_mismatch", 0)
        for label, val, threshold in [
            ("Accuracy", acc, 0.83), ("Macro F1", f1, 0.82),
            ("Recall Consistent", rc, 0.78), ("Recall Mismatch", rm, 0.78),
        ]:
            colour = "green" if val >= threshold else "red"
            st.markdown(f"<span style='color:{colour}'>●</span> **{label}:** {val:.3f}", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: load cached predictions (for dashboard / heatmap)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_predictions():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pred_path = os.path.join(base, "outputs", "predictions.csv")
    if os.path.exists(pred_path):
        return pd.read_csv(pred_path)
    # Fall back to pseudo-labeled
    pseudo_path = os.path.join(base, "outputs", "pseudo_labeled.csv")
    if os.path.exists(pseudo_path):
        df = pd.read_csv(pseudo_path)
        df["predicted_mismatch"] = df["mismatch_label"]
        df["mismatch_probability"] = 0.75
        return df
    return None


@st.cache_data(ttl=300)
def load_dossiers():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dos_path = os.path.join(base, "outputs", "dossiers.json")
    if os.path.exists(dos_path):
        with open(dos_path) as f:
            return json.load(f)
    return []


# ---------------------------------------------------------------------------
# Helper: run quick inference on a single ticket (rule-based fast path)
# ---------------------------------------------------------------------------
def quick_infer_single(ticket_dict: dict) -> dict:
    """
    Fast CPU inference using rule-based signals only (no model load in app).
    Returns prediction dict.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pseudo_labeler import (
        _rule_score, PRIORITY_MAP, INV_PRIORITY_MAP, _score_to_priority
    )
    from dossier_generator import extract_ticket_evidence, build_feature_evidence, \
        build_constraint_analysis_local, estimate_confidence

    row = pd.Series(ticket_dict)
    sa = _rule_score(
        str(ticket_dict.get("Ticket_Subject", "")),
        str(ticket_dict.get("Ticket_Description", ""))
    )
    # Simple resolution time signal
    rt = float(ticket_dict.get("Resolution_Time_Hours", 24) or 24)
    sb = min(rt / 40.0, 1.0) * 3.0

    fused = 0.60 * sa + 0.40 * sb
    inferred = _score_to_priority(fused)
    assigned = ticket_dict.get("Priority_Level", "Medium")
    assigned_ord = PRIORITY_MAP.get(assigned, 1)
    inferred_ord = PRIORITY_MAP.get(inferred, 1)
    delta = inferred_ord - assigned_ord
    mismatch = int(abs(delta) >= 1)
    mismatch_type = "Hidden Crisis" if delta > 0 else ("False Alarm" if delta < 0 else "Consistent")

    row["inferred_severity"] = inferred
    row["mismatch_label"] = mismatch
    row["mismatch_type"] = mismatch_type
    row["severity_delta"] = delta
    row["score_a"] = sa
    row["score_b"] = sb
    row["score_c"] = sa * 0.8  # proxy
    row["fused_score"] = fused

    ev = extract_ticket_evidence(row)
    feat_ev = build_feature_evidence(ev, assigned, inferred, sa)
    constraint = build_constraint_analysis_local(ev, assigned, inferred, mismatch_type)
    conf = estimate_confidence(fused, sa, sb, sa * 0.8, mismatch)

    dossier = {
        "ticket_id": str(ticket_dict.get("Ticket_ID", "SINGLE-001")),
        "assigned_priority": assigned,
        "inferred_severity": inferred,
        "mismatch_type": mismatch_type if mismatch else "Consistent",
        "severity_delta": f"{delta:+d}",
        "feature_evidence": feat_ev,
        "constraint_analysis": constraint,
        "confidence": conf,
        "is_mismatch": bool(mismatch),
    }
    return dossier


# ---------------------------------------------------------------------------
# Helper: render dossier card
# ---------------------------------------------------------------------------
def render_dossier(d: dict):
    mtype = d.get("mismatch_type", "Consistent")
    if mtype == "Hidden Crisis":
        badge = '<span class="crisis-badge">🚨 Hidden Crisis</span>'
    elif mtype == "False Alarm":
        badge = '<span class="alarm-badge">⚠️ False Alarm</span>'
    else:
        badge = '<span class="consistent-badge">✅ Consistent</span>'

    cols = st.columns([2, 1, 1, 1])
    with cols[0]:
        st.markdown(f"**Ticket:** `{d.get('ticket_id', 'N/A')}`")
        st.markdown(badge, unsafe_allow_html=True)
    with cols[1]:
        st.metric("Assigned", d.get("assigned_priority", "—"))
    with cols[2]:
        st.metric("Inferred", d.get("inferred_severity", "—"))
    with cols[3]:
        delta_str = str(d.get("severity_delta", "0"))
        delta_val = int(delta_str.replace("+", "")) if delta_str not in ("", "None") else 0
        st.metric("Δ Severity", delta_str, delta=delta_val if delta_val != 0 else None)

    st.markdown("**Evidence:**")
    for ev in d.get("feature_evidence", []):
        sig = ev.get("signal", "")
        val = ev.get("value", "")
        interp = ev.get("interpretation", ev.get("weight", ""))
        field = ev.get("field", "")
        st.markdown(
            f'<div class="evidence-block">'
            f'<b>{sig}</b> <code>[{field}]</code> → <i>{val}</i>'
            f'{f" — {interp}" if interp else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("**Constraint Analysis:**")
    st.info(d.get("constraint_analysis", "N/A"))
    st.caption(f"Confidence: {d.get('confidence', 0):.1%}")


# ===========================================================================
# PAGE 1: Single Ticket Audit
# ===========================================================================
if page == "🎫 Single Ticket Audit":
    st.markdown('<p class="main-title">🔍 Single Ticket Audit</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Submit a support ticket to get an instant mismatch judgment and Evidence Dossier.</p>', unsafe_allow_html=True)

    with st.form("ticket_form"):
        c1, c2 = st.columns(2)
        with c1:
            ticket_id     = st.text_input("Ticket ID", value="TKT-TEST-001")
            ticket_subject = st.text_input("Ticket Subject", value="Application crashes on startup")
            issue_category = st.selectbox("Issue Category",
                ["Technical", "Billing", "Account", "General Inquiry", "Fraud"])
            ticket_channel = st.selectbox("Ticket Channel", ["Email", "Chat", "Web Form"])
        with c2:
            priority_level = st.selectbox("Assigned Priority",
                ["Low", "Medium", "High", "Critical"])
            resolution_time = st.number_input("Resolution Time (hours)", min_value=1, max_value=500, value=48)
            satisfaction    = st.slider("Satisfaction Score (1-5)", 1, 5, 3)
            customer_email  = st.text_input("Customer Email", value="user@example.com")

        ticket_description = st.text_area(
            "Ticket Description",
            height=120,
            value="The application crashes every time I open the settings tab. "
                  "We cannot access critical data and have a deadline tomorrow. "
                  "This is affecting our entire team of 50 users."
        )

        submitted = st.form_submit_button("🔍 Audit This Ticket", type="primary", use_container_width=True)

    if submitted:
        ticket_dict = {
            "Ticket_ID": ticket_id,
            "Ticket_Subject": ticket_subject,
            "Ticket_Description": ticket_description,
            "Priority_Level": priority_level,
            "Issue_Category": issue_category,
            "Ticket_Channel": ticket_channel,
            "Resolution_Time_Hours": resolution_time,
            "Satisfaction_Score": satisfaction,
            "Customer_Email": customer_email,
        }

        with st.spinner("Auditing ticket..."):
            dossier = quick_infer_single(ticket_dict)

        if dossier["is_mismatch"]:
            if dossier["mismatch_type"] == "Hidden Crisis":
                st.error(f"🚨 **PRIORITY MISMATCH DETECTED — Hidden Crisis**  \nThis ticket appears more severe than its assigned priority.")
            else:
                st.warning(f"⚠️ **PRIORITY MISMATCH DETECTED — False Alarm**  \nThis ticket appears less severe than its assigned priority.")
        else:
            st.success("✅ **No mismatch detected.** Priority appears consistent with ticket characteristics.")

        st.divider()
        st.markdown("### 📋 Evidence Dossier")
        render_dossier(dossier)

        st.divider()
        st.markdown("### 📄 Raw Dossier JSON")
        st.json(dossier)


# ===========================================================================
# PAGE 2: Batch CSV Upload
# ===========================================================================
elif page == "📂 Batch CSV Upload":
    st.markdown('<p class="main-title">📂 Batch CSV Audit</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a CSV of support tickets for bulk mismatch detection.</p>', unsafe_allow_html=True)

    required_cols = ["Ticket_Subject", "Ticket_Description", "Priority_Level",
                     "Issue_Category", "Ticket_Channel", "Resolution_Time_Hours"]

    st.info(f"**Required columns:** {', '.join(required_cols)}")

    uploaded = st.file_uploader("Upload tickets CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df):,} tickets")
        st.dataframe(df.head(5), use_container_width=True)

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            if st.button("🚀 Run Batch Audit", type="primary"):
                with st.spinner(f"Auditing {len(df):,} tickets... (rule-based fast path)"):
                    from pseudo_labeler import (
                        compute_signal_a, compute_signal_b, compute_signal_c,
                        fuse_signals, compute_mismatch_label, PRIORITY_MAP
                    )
                    sa = compute_signal_a(df)
                    sb = compute_signal_b(df, signal_a_scores=sa)
                    sc_ = compute_signal_c(df)
                    fused, inferred = fuse_signals(sa, sb, sc_)

                    assigned_ord = np.array([PRIORITY_MAP.get(p, 1) for p in df["Priority_Level"]])
                    inferred_ord = np.array([PRIORITY_MAP.get(p, 1) for p in inferred])
                    delta = inferred_ord - assigned_ord
                    mismatch = (np.abs(delta) >= 1).astype(int)

                    df["inferred_severity"] = inferred
                    df["predicted_mismatch"] = mismatch
                    df["severity_delta"] = delta
                    df["mismatch_type"] = np.where(
                        delta > 0, "Hidden Crisis",
                        np.where(delta < 0, "False Alarm", "Consistent")
                    )
                    df["fused_score"] = fused
                    df["score_a"] = sa

                # Summary
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Tickets", f"{len(df):,}")
                col2.metric("Mismatches Flagged", f"{int(mismatch.sum()):,}")
                col3.metric("Hidden Crisis", f"{int((df['mismatch_type']=='Hidden Crisis').sum()):,}")
                col4.metric("False Alarm", f"{int((df['mismatch_type']=='False Alarm').sum()):,}")

                st.dataframe(
                    df[["Ticket_ID" if "Ticket_ID" in df.columns else df.columns[0],
                        "Priority_Level", "inferred_severity", "mismatch_type",
                        "severity_delta", "predicted_mismatch"]].head(50),
                    use_container_width=True
                )

                # Download
                csv_out = df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Predictions CSV",
                    data=csv_out,
                    file_name="sia_predictions.csv",
                    mime="text/csv",
                )

                # Quick dossiers for flagged
                from dossier_generator import generate_dossiers_batch
                flagged = df[df["predicted_mismatch"] == 1].copy()
                if len(flagged) > 0:
                    dossiers = generate_dossiers_batch(flagged.head(20))
                    dossier_json = json.dumps(dossiers, indent=2)
                    st.download_button(
                        "⬇️ Download Evidence Dossiers JSON",
                        data=dossier_json,
                        file_name="sia_dossiers.json",
                        mime="application/json",
                    )


# ===========================================================================
# PAGE 3: Priority Mismatch Dashboard
# ===========================================================================
elif page == "📊 Mismatch Dashboard":
    st.markdown('<p class="main-title">📊 Priority Mismatch Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Distribution of flagged tickets, mismatch types, and top contributing signals.</p>', unsafe_allow_html=True)

    df = load_predictions()
    if df is None:
        st.warning("No predictions found. Run the training pipeline first or upload a CSV.")
        st.stop()

    mismatch_col = "predicted_mismatch" if "predicted_mismatch" in df.columns else "mismatch_label"
    total = len(df)
    flagged = int(df[mismatch_col].sum())
    consistent = total - flagged

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tickets", f"{total:,}")
    c2.metric("Mismatches Flagged", f"{flagged:,}", delta=f"{flagged/total:.1%}")
    c3.metric("Hidden Crisis",
              f"{int((df['mismatch_type']=='Hidden Crisis').sum()):,}" if 'mismatch_type' in df.columns else "—")
    c4.metric("False Alarm",
              f"{int((df['mismatch_type']=='False Alarm').sum()):,}" if 'mismatch_type' in df.columns else "—")

    st.divider()
    row1_c1, row1_c2 = st.columns(2)

    # Pie: flagged vs consistent
    with row1_c1:
        fig_pie = px.pie(
            values=[consistent, flagged],
            names=["Consistent", "Mismatch"],
            color_discrete_sequence=["#28a745", "#ff4d4d"],
            title="Overall Mismatch Rate",
            hole=0.4,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Bar: mismatch type breakdown
    with row1_c2:
        if "mismatch_type" in df.columns:
            type_counts = df["mismatch_type"].value_counts().reset_index()
            type_counts.columns = ["Mismatch Type", "Count"]
            fig_bar = px.bar(
                type_counts, x="Mismatch Type", y="Count",
                color="Mismatch Type",
                color_discrete_map={
                    "Hidden Crisis": "#ff4d4d",
                    "False Alarm": "#ff9500",
                    "Consistent": "#28a745"
                },
                title="Mismatch Type Distribution",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    row2_c1, row2_c2 = st.columns(2)

    # Mismatch rate by priority level
    with row2_c1:
        if mismatch_col in df.columns and "Priority_Level" in df.columns:
            pr_mismatch = df.groupby("Priority_Level")[mismatch_col].mean().reset_index()
            pr_mismatch.columns = ["Priority Level", "Mismatch Rate"]
            pr_mismatch["Mismatch Rate (%)"] = pr_mismatch["Mismatch Rate"] * 100
            order = ["Low", "Medium", "High", "Critical"]
            pr_mismatch["Priority Level"] = pd.Categorical(pr_mismatch["Priority Level"], categories=order, ordered=True)
            pr_mismatch = pr_mismatch.sort_values("Priority Level")
            fig_pr = px.bar(
                pr_mismatch, x="Priority Level", y="Mismatch Rate (%)",
                color="Mismatch Rate (%)", color_continuous_scale="RdYlGn_r",
                title="Mismatch Rate by Assigned Priority",
            )
            st.plotly_chart(fig_pr, use_container_width=True)

    # Top contributing signals (from ablation)
    with row2_c2:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abl_path = os.path.join(base_dir, "outputs", "ablation.json")
        if os.path.exists(abl_path):
            with open(abl_path) as f:
                abl = json.load(f)
            signals = abl.get("individual_signal_mismatch_rate", {})
            if signals:
                sig_df = pd.DataFrame({
                    "Signal": ["Rule-based NLP\n(Signal A)", "Resolution Time\n(Signal B)", "Embedding Cluster\n(Signal C)"],
                    "Mismatch Rate": [
                        signals.get("signal_a_only", 0),
                        signals.get("signal_b_only", 0),
                        signals.get("signal_c_only", 0),
                    ]
                })
                fig_sig = px.bar(
                    sig_df, x="Signal", y="Mismatch Rate",
                    color="Signal", title="Signal Ablation: Mismatch Rate per Signal",
                    color_discrete_sequence=["#4361ee", "#7209b7", "#f72585"],
                )
                fig_sig.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig_sig, use_container_width=True)

    # Mismatch by category and channel
    st.divider()
    row3_c1, row3_c2 = st.columns(2)

    with row3_c1:
        if "Issue_Category" in df.columns:
            cat_m = df.groupby("Issue_Category")[mismatch_col].mean().sort_values(ascending=True).reset_index()
            cat_m.columns = ["Category", "Mismatch Rate"]
            fig_cat = px.bar(cat_m, x="Mismatch Rate", y="Category", orientation="h",
                             title="Mismatch Rate by Issue Category",
                             color="Mismatch Rate", color_continuous_scale="Reds")
            fig_cat.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig_cat, use_container_width=True)

    with row3_c2:
        if "Ticket_Channel" in df.columns:
            ch_m = df.groupby("Ticket_Channel")[mismatch_col].mean().reset_index()
            ch_m.columns = ["Channel", "Mismatch Rate"]
            fig_ch = px.pie(ch_m, values="Mismatch Rate", names="Channel",
                            title="Mismatch Rate by Channel", hole=0.3)
            st.plotly_chart(fig_ch, use_container_width=True)


# ===========================================================================
# PAGE 4: Severity Delta Heatmap
# ===========================================================================
elif page == "🌡️ Severity Heatmap":
    st.markdown('<p class="main-title">🌡️ Severity Delta Heatmap</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Average severity delta (inferred − assigned) by Issue Category × Ticket Channel.</p>', unsafe_allow_html=True)

    df = load_predictions()
    if df is None:
        st.warning("No predictions found. Run the training pipeline first.")
        st.stop()

    if "severity_delta" not in df.columns:
        st.warning("severity_delta column not found in predictions.")
        st.stop()

    if "Issue_Category" in df.columns and "Ticket_Channel" in df.columns:
        pivot = df.pivot_table(
            values="severity_delta",
            index="Issue_Category",
            columns="Ticket_Channel",
            aggfunc="mean",
        )
        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Avg Severity Delta"),
            text=[[f"{v:.2f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 13},
        ))
        fig_heat.update_layout(
            title="Severity Delta Heatmap: Issue Category × Ticket Channel",
            xaxis_title="Ticket Channel",
            yaxis_title="Issue Category",
            height=450,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("""
        **How to read this:**
        - **Red (positive Δ):** Tickets in this bucket are systematically *under-triaged* — true severity is higher than assigned.
        - **Blue (negative Δ):** Tickets are *over-triaged* — assigned priority exceeds true severity.
        - **White (≈0):** Assignments are well-calibrated for this combination.
        """)

    # Also: distribution of severity deltas
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(
            df, x="severity_delta", nbins=7,
            color_discrete_sequence=["#4361ee"],
            title="Distribution of Severity Deltas",
            labels={"severity_delta": "Severity Delta (Inferred − Assigned)"},
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No mismatch")
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        if "Priority_Level" in df.columns:
            box_fig = px.box(
                df, x="Priority_Level", y="severity_delta",
                color="Priority_Level",
                category_orders={"Priority_Level": ["Low", "Medium", "High", "Critical"]},
                title="Severity Delta by Assigned Priority",
                color_discrete_sequence=["#28a745", "#ffc107", "#ff9500", "#dc3545"],
            )
            box_fig.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig_hist if True else box_fig, use_container_width=True)
            st.plotly_chart(box_fig, use_container_width=True)
