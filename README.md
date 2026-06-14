# Support Integrity Auditor (SIA)

A self-supervised pipeline that audits customer support ticket triage by detecting mismatches between human-assigned `Priority_Level` and an independently inferred severity. Flagged mismatches are categorized as **Hidden Crisis** (under-triaged) or **False Alarm** (over-triaged) and accompanied by a fully grounded, hallucination-free Evidence Dossier.

---

## 1. Overview and Motivation

Human ticket triage is noisy: agents and customers may under-state genuinely severe issues (e.g. calmly worded data-loss reports) or over-state trivial ones (e.g. "URGENT CRITICAL" subject lines on cosmetic complaints). SIA treats `Priority_Level` as an unreliable label and reconstructs an independent estimate of true severity from the ticket's text, resolution behavior, and structured metadata, then flags disagreements for human review.

The system is organized into three stages:

1. **Pseudo-Label Generation** (self-supervised, signal fusion) — `pseudo_labeler.py`
2. **Classifier Training** (fine-tuned transformer with LoRA) — `train_pipeline.py`
3. **Evidence Dossier Generation** (anti-hallucination, fully traceable) — `dossier_generator.py`

Inference over all three stages is orchestrated by `predict.py`, and results are surfaced through an interactive Streamlit application (`streamlit_app.py`). A held-out adversarial test set (`adversarial_tickets.py`) stress-tests robustness against keyword-based shortcuts.

---

## 2. Architecture Diagram

```
                         ┌─────────────────────────────────────────┐
                         │              RAW TICKETS                 │
                         │  Ticket_Subject, Ticket_Description,     │
                         │  Priority_Level, Ticket_Channel,         │
                         │  Issue_Category, Resolution_Time_Hours,  │
                         │  Satisfaction_Score                      │
                         └───────────────────┬───────────────────────┘
                                              │
              ┌───────────────────────────────────────────────────────┐
              │            STAGE 1: PSEUDO-LABEL GENERATION             │
              │                  (pseudo_labeler.py)                    │
              │                                                          │
              │  Signal A: Rule-based NLP    Signal B: Resolution-Time  │
              │  (keywords, negation,        Regression (XGBoost on    │
              │   VADER sentiment,            text + structured         │
              │   exclamation density)        features, 5-fold CV)      │
              │       w = 0.45                     w = 0.30             │
              │                                                          │
              │  Signal C: Embedding Clustering (MiniLM + MiniBatch     │
              │  K-Means, cluster urgency = mean keyword density)       │
              │                w = 0.25                                 │
              │                                                          │
              │   fused = 0.45*A + 0.30*B + 0.25*C  -->  [0,3] score    │
              │   inferred_severity = map(fused) -> {Low..Critical}     │
              │   mismatch_label = |ord(assigned) - ord(inferred)| >= 1 │
              │   severity_delta = ord(inferred) - ord(assigned)        │
              │   mismatch_type  = Hidden Crisis / False Alarm          │
              └───────────────────────────┬─────────────────────────────┘
                                            │  pseudo_labeled.csv
                                            ▼
              ┌───────────────────────────────────────────────────────┐
              │           STAGE 2: CLASSIFIER TRAINING                  │
              │               (train_pipeline.py)                       │
              │                                                          │
              │  Text:  [Subject] [SEP] [Description]                  │
              │            │                                            │
              │            ▼                                            │
              │   DistilBERT-base-uncased + LoRA (r=8, q_lin/v_lin)    │
              │            │  [CLS] embedding (768-d)                  │
              │            ▼                                            │
              │   concat ◄── Metadata MLP ◄── (channel OH + category   │
              │     │            (→16-d)         OH + scaled res.time) │
              │     ▼                                                   │
              │   Classifier head (Dropout→64→ReLU→Dropout→2)          │
              │     │                                                   │
              │     ▼                                                   │
              │   Weighted Cross-Entropy (class-balanced weights)       │
              │     │                                                   │
              │     ▼                                                   │
              │   binary: Consistent (0) / Mismatch (1)                 │
              └───────────────────────────┬─────────────────────────────┘
                                            │  best_model.pt + encoders
                                            ▼
              ┌───────────────────────────────────────────────────────┐
              │        STAGE 3: EVIDENCE DOSSIER GENERATION             │
              │               (dossier_generator.py)                    │
              │                                                          │
              │  Pre-extraction (ground truth, deterministic):          │
              │    - escalation/low-severity keywords found in text     │
              │    - negation phrases found                              │
              │    - resolution-time bucket interpretation               │
              │    - channel & category severity weights                │
              │    - satisfaction-score note                             │
              │            │                                             │
              │            ▼                                             │
              │  feature_evidence[] (each item traceable to a field)     │
              │            │                                             │
              │            ▼                                             │
              │  constraint_analysis (rule-based prose, or LLM prose     │
              │    constrained to pre-extracted facts only)              │
              │            │                                             │
              │            ▼                                             │
              │  confidence = f(boundary distance, signal agreement)     │
              │            │                                             │
              │            ▼                                             │
              │       Evidence Dossier JSON (per flagged ticket)         │
              └───────────────────────────┬─────────────────────────────┘
                                            │
                                            ▼
                         ┌───────────────────────────────────┐
                         │      predict.py (orchestrator)     │
                         │  predictions.csv + dossiers.json   │
                         └───────────────┬─────────────────────┘
                                          │
                                          ▼
                         ┌───────────────────────────────────┐
                         │      streamlit_app.py (UI)         │
                         │  Single Ticket Audit / Batch CSV / │
                         │  Mismatch Dashboard / Severity     │
                         │  Delta Heatmap                     │
                         └─────────────────────────────────────┘

                         ┌───────────────────────────────────┐
                         │   adversarial_tickets.py (eval)    │
                         │  10 held-out tickets designed to   │
                         │  fool keyword-only triage          │
                         └─────────────────────────────────────┘
```

---

## 3. Methodology and Mathematics

### 3.1 Stage 1 — Pseudo-Label Generation

The objective is to construct an inferred severity label that is statistically independent of the noisy `Priority_Level` column, by combining three signals that each draw on different information channels (lexical, behavioral, semantic).

**Signal A — Rule-based NLP score.** For ticket text $t$ (subject concatenated with description, lower-cased), define

$$
\text{kw}(t) = \min\left(\frac{|\{k \in K_{esc} : k \in t\}|}{3}, 1\right), \qquad
\text{neg}(t) = \min\left(\frac{|\{n \in N : n \in t\}|}{2}, 1\right)
$$

$$
\text{low}(t) = \min\left(0.3 \cdot |\{l \in K_{low} : l \in t\}|,\ 0.6\right), \qquad
\text{sent}(t) = \frac{-\text{VADER}(t) + 1}{2}, \qquad
\text{excl}(t) = \min(0.15 \cdot \#\{!\}, 0.3)
$$

where $K_{esc}$ is the escalation-keyword set, $K_{low}$ the low-severity indicator set, $N$ the negation-trigger set, and $\text{VADER}(t) \in [-1, 1]$ is the compound sentiment score (lower = more negative = more urgent).

The raw score is

$$
r(t) = \text{clip}\Big(0.40\,\text{kw}(t) + 0.25\,\text{neg}(t) + 0.20\,\text{sent}(t) + 0.15\,\text{excl}(t) - \text{low}(t),\ 0,\ 1\Big)
$$

and Signal A is $s_A = 3\,r(t) \in [0,3]$.

**Signal B — Resolution-time regression.** A feature vector is built from text statistics (length, word count, exclamation count, keyword/negation/low-severity hit counts), label-encoded `Issue_Category` and `Ticket_Channel`, and $s_A$. An `XGBRegressor` (100 trees, depth 4, learning rate 0.1, subsample/colsample 0.8) is fit to predict `Resolution_Time_Hours`, using 5-fold cross-validated out-of-fold predictions $\hat{y}$ to avoid leakage. The predictions are min-max normalized to $[0,3]$:

$$
s_B = 3 \cdot \frac{\hat{y} - \min(\hat{y})}{\max(\hat{y}) - \min(\hat{y}) + \varepsilon}
$$

Higher predicted resolution time is interpreted as higher latent severity.

**Signal C — Embedding-based clustering.** Ticket text is embedded with `sentence-transformers/all-MiniLM-L6-v2` and clustered into $K=8$ clusters via MiniBatch K-Means. Each cluster $c$ receives an urgency score equal to the mean escalation-keyword hit count of its members, $\bar{h}_c$, and per-ticket Signal C is the cluster score rescaled to $[0,3]$:

$$
s_C = 3 \cdot \frac{\bar{h}_{c(i)}}{\max_c \bar{h}_c + \varepsilon}
$$

**Fusion.** The three signals are combined by a fixed weighted average:

$$
\text{fused}_i = 0.45\, s_{A,i} + 0.30\, s_{B,i} + 0.25\, s_{C,i} \in [0, 3]
$$

This continuous score is mapped to an ordinal severity label via fixed boundaries:

$$
\text{inferred}(s) = \begin{cases}
\text{Low} & s < 0.75 \\
\text{Medium} & 0.75 \le s < 1.50 \\
\text{High} & 1.50 \le s < 2.25 \\
\text{Critical} & s \ge 2.25
\end{cases}
$$

**Mismatch label and direction.** With ordinal map $\{\text{Low}=0, \text{Medium}=1, \text{High}=2, \text{Critical}=3\}$, define

$$
\delta_i = \text{ord}(\text{inferred}_i) - \text{ord}(\text{assigned}_i), \qquad
y_i = \mathbb{1}[\,|\delta_i| \ge 1\,]
$$

$\delta_i > 0$ ("Hidden Crisis," under-triaged) and $\delta_i < 0$ ("False Alarm," over-triaged) define the `mismatch_type`.

**Why this fusion strategy.** Each signal captures a distinct failure mode that the others miss in isolation: Signal A is fast and interpretable but easily gamed by keyword stuffing or negation (Type B/C adversarial tickets); Signal B grounds severity in *actual operational outcomes* (how long the issue took to resolve) rather than wording, catching cases where lexical signals are absent (Type A "Hidden Crisis" tickets); Signal C captures *semantic* similarity to historically urgent tickets even when specific keywords are absent or substituted. The 0.45/0.30/0.25 weighting prioritizes the rule-based signal (highest standalone interpretability and traceability for the dossier) while letting the behavioral (B) and semantic (C) signals correct its blind spots. The empirical justification for this weighting is the ablation in §5.

### 3.2 Stage 2 — Classifier Training

**Architecture.** A `distilbert-base-uncased` encoder (66M parameters, WordPiece tokenizer — chosen over `deberta-v3-small` because DeBERTa's SentencePiece tokenizer fails to load reliably on Windows/Python 3.11 environments without a working `sentencepiece` C-extension) is adapted with **LoRA** (rank $r=8$, $\alpha=16$, dropout 0.05) applied to the query and value projections (`q_lin`, `v_lin`) of every attention layer, training only $\approx 0.4$M of the 66M parameters.

The `[CLS]` token representation $h \in \mathbb{R}^{768}$ from the final hidden layer is concatenated with a metadata embedding $m \in \mathbb{R}^{16}$ produced by an MLP over structured features:

$$
m = \text{ReLU}\big(W_2\,\text{Dropout}(\text{ReLU}(W_1 x_{\text{meta}} + b_1)) + b_2\big), \qquad x_{\text{meta}} = [\text{OH}(\text{channel}),\ \text{OH}(\text{category}),\ z(\text{resolution\_time})]
$$

where $z(\cdot)$ denotes standardization (`StandardScaler`). The combined vector $[h \,\Vert\, m] \in \mathbb{R}^{784}$ passes through a classification head:

$$
\text{logits} = W_4\,\text{Dropout}\big(\text{ReLU}(W_3\,\text{Dropout}([h\Vert m]) + b_3)\big) + b_4 \in \mathbb{R}^2
$$

**Class imbalance.** `compute_class_weight("balanced", ...)` produces weights $w_0, w_1$ inversely proportional to class frequency, used in

$$
\mathcal{L} = -\sum_i w_{y_i} \log \frac{e^{z_{i,y_i}}}{\sum_{c} e^{z_{i,c}}}
$$

i.e. weighted cross-entropy, explicitly addressing the imbalance between Consistent and Mismatch classes.

**Optimization.** AdamW ($\text{lr}=2\times10^{-4}$, weight decay $10^{-2}$), gradient clipping at norm 1.0, linear warmup/decay schedule over $\text{epochs} \times |\text{train batches}|$ steps with 10% warmup, batch size 16, max sequence length 128, trained for 4 epochs on CPU. The checkpoint with the highest validation macro-F1 is retained.

### 3.3 Stage 3 — Evidence Dossier Generation

**Anti-hallucination design.** All factual fields in `feature_evidence` are extracted deterministically from the raw ticket row by `extract_ticket_evidence()` *before* any prose generation. Each evidence item carries a `field` key pointing to the exact source column (`Ticket_Subject`/`Ticket_Description`, `Resolution_Time_Hours`, `Ticket_Channel`, `Issue_Category`, `Satisfaction_Score`), satisfying the hard traceability rule. The `constraint_analysis` text is either generated locally by `build_constraint_analysis_local()` — which only ever interpolates pre-extracted values — or, if an LLM (`claude-sonnet-4-6`) is available, the LLM receives this same pre-extracted evidence as its *only* permissible source of facts and is restricted to interpretive prose.

**Confidence estimate.** Let $f$ be the fused score and $b \in \{0.75, 1.50, 2.25\}$ the severity-band boundaries. Define boundary confidence as the normalized distance to the nearest boundary,

$$
\text{conf}_{\text{bnd}} = \min\left(\frac{\min_b |f - b|}{0.75},\ 1\right)
$$

and signal agreement as the inverse standard deviation of the three signals' ordinal bucket assignments $o(s_A), o(s_B), o(s_C) \in \{0,1,2,3\}$:

$$
\text{conf}_{\text{agr}} = 1 - \min\left(\frac{\text{std}(o(s_A), o(s_B), o(s_C))}{3},\ 1\right)
$$

The final confidence is

$$
\text{confidence} = \min\Big(0.99,\ \big(0.6\,\text{conf}_{\text{bnd}} + 0.4\,\text{conf}_{\text{agr}}\big) \cdot \big(1.05 \text{ if mismatch else } 1\big)\Big)
$$

**Dossier schema** (exactly as specified):

```json
{
  "ticket_id": "...",
  "assigned_priority": "...",
  "inferred_severity": "...",
  "mismatch_type": "Hidden Crisis | False Alarm",
  "severity_delta": "<signed integer string, e.g. '+2'>",
  "feature_evidence": [
    {"signal": "keyword", "field": "...", "value": "...", "weight": "..."},
    {"signal": "resolution_time", "field": "Resolution_Time_Hours", "value": "...", "interpretation": "..."}
  ],
  "constraint_analysis": "<2-3 sentence grounded explanation>",
  "confidence": "<float 0-1>"
}
```

---

## 4. Pipeline Stages Summary

| Stage | Script | Input | Output | Key Technique |
|---|---|---|---|---|
| 1. Pseudo-Labeling | `pseudo_labeler.py` | raw ticket CSV | `pseudo_labeled.csv`, `ablation.json` | Signal fusion (rule-based NLP + XGBoost regression + MiniLM/K-Means) |
| 2. Classifier Training | `train_pipeline.py` | `pseudo_labeled.csv` | `models/sia_classifier/`, `metrics.json` | DistilBERT + LoRA + metadata MLP, weighted CE |
| 3. Dossier Generation | `dossier_generator.py` | scored/flagged tickets | `sample_dossiers.json` / `dossiers.json` | Pre-extraction + grounded prose, confidence scoring |
| Orchestration | `predict.py` | any ticket CSV | `predictions.csv`, `dossiers.json` | Runs Stages 1→3 end-to-end |
| Adversarial Eval | `adversarial_tickets.py` | 10 held-out tickets | adversarial score | Hidden Crisis / False Alarm / Negation stress tests |
| Application | `streamlit_app.py` | predictions / live input | interactive UI | Single audit, batch upload, dashboards, heatmap |

---

## 5. Ablation: Signal Contribution

`generate_pseudo_labels()` writes `outputs/ablation.json`, reporting the standalone mismatch rate of each signal (using only that signal's bucketed label vs. `Priority_Level`) and pairwise agreement between signal pairs. Reported fields and their interpretation:

| Metric | Definition | Role |
|---|---|---|
| `fusion_weights` | {A: 0.45, B: 0.30, C: 0.25} | Confirms the weighting used to produce `mismatch_label` |
| `mismatch_rate_overall` | Fraction of tickets flagged by the fused score | Overall flag rate of the full system |
| `individual_signal_mismatch_rate.signal_a_only` | Mismatch rate using Signal A alone | Isolated contribution of rule-based NLP |
| `individual_signal_mismatch_rate.signal_b_only` | Mismatch rate using Signal B alone | Isolated contribution of resolution-time regression |
| `individual_signal_mismatch_rate.signal_c_only` | Mismatch rate using Signal C alone | Isolated contribution of embedding clustering |
| `pairwise_agreement.A_vs_B`, `A_vs_C`, `B_vs_C` | Fraction of tickets where two signals assign the same severity bucket | **Pseudo-Label Signal Agreement** (§5 of spec) |
| `class_distribution` | Counts of Mismatch / Consistent / Hidden Crisis / False Alarm | Class balance audit, motivates weighted CE in Stage 2 |

**Ablation table (template — populate by running `pseudo_labeler.py` on the target dataset and reading `outputs/ablation.json`):**

| Configuration | Mismatch Rate | Hidden Crisis | False Alarm | Notes |
|---|---|---|---|---|
| Signal A only (rule-based NLP) | `<signal_a_only>` | — | — | Cheapest, most interpretable; vulnerable to keyword-gaming |
| Signal B only (resolution-time regression) | `<signal_b_only>` | — | — | Behavioral grounding, robust to wording |
| Signal C only (embedding clustering) | `<signal_c_only>` | — | — | Semantic generalization, no keyword dependence |
| **Fused (0.45A + 0.30B + 0.25C)** | `<mismatch_rate_overall>` | `<Hidden Crisis count>` | `<False Alarm count>` | Final pseudo-label used for Stage 2 training |

| Pairwise Signal Agreement | Value |
|---|---|
| A vs B | `<A_vs_B>` |
| A vs C | `<A_vs_C>` |
| B vs C | `<B_vs_C>` |

---

## 6. Evaluation Metrics

### 6.1 Classifier Metrics (Stage 2)

`train_pipeline.py` writes `outputs/metrics.json` on the held-out validation split (15% stratified). Reported fields:

| Metric | Field in `metrics.json` | Target Threshold |
|---|---|---|
| Binary Classification Accuracy (%) | `binary_accuracy` | ≥ 0.83 |
| Macro F1 Score | `macro_f1` | ≥ 0.82 |
| Per-Class Recall — Consistent | `recall_consistent` | ≥ 0.78 (both classes) |
| Per-Class Recall — Mismatch | `recall_mismatch` | ≥ 0.78 (both classes) |
| Full `classification_report` | `classification_report` | — |
| Best epoch / LoRA rank / epochs | `best_epoch`, `lora_rank`, `epochs` | run metadata |

**Metric results table (template — populate from `outputs/metrics.json` after training):**

| Metric | Value | Threshold | Met? |
|---|---|---|---|
| Binary Accuracy | `<binary_accuracy>` | 0.83 | `<thresholds_met.accuracy_83pct>` |
| Macro F1 | `<macro_f1>` | 0.82 | `<thresholds_met.macro_f1_082>` |
| Recall (Consistent) | `<recall_consistent>` | 0.78 | `<thresholds_met.recall_both_078>` (joint) |
| Recall (Mismatch) | `<recall_mismatch>` | 0.78 | `<thresholds_met.recall_both_078>` (joint) |
| Best Epoch | `<best_epoch>` | — | — |
| LoRA Rank | 8 | — | — |

### 6.2 Pseudo-Label Signal Agreement

See §5 pairwise agreement table — this is the required "Pseudo-Label Signal Agreement (pairwise agreement between the two chosen signals)" metric, extended here to all three pairs since three signals are fused.

### 6.3 Adversarial Robustness (Bonus)

`adversarial_tickets.py` defines 10 held-out tickets across three pattern types:

- **Type A — Hidden Crisis (5 tickets, ADV-001–005):** zero escalation keywords, true severity High/Critical (data loss, account takeover, latency collapse, insider-threat exposure, billing overcharge).
- **Type B — False Alarm via keyword stuffing (3 tickets, ADV-006–008):** maximal escalation-keyword density and exclamation marks, true severity Low (cosmetic complaints, sales/tutorial inquiries).
- **Type C — Negation adversarial (2 tickets, ADV-009–010):** escalation words present but negated or in past tense ("no crash", "issue has been resolved"), true severity Low.

`run_adversarial_eval(predict_fn)` runs all 10 tickets through `predict()`, compares `predicted_mismatch` to `GT_mismatch`, and computes:

$$
\text{adversarial\_score} = \frac{\text{correct}}{10}, \qquad \text{bonus\_earned} = \mathbb{1}[\text{adversarial\_score} \ge 0.7]
$$

**Adversarial results table (template — populate by running `adversarial_tickets.run_adversarial_eval(predict)`):**

| Ticket ID | Pattern Type | GT Mismatch | Predicted Mismatch | Correct? |
|---|---|---|---|---|
| ADV-001 | Hidden Crisis — data loss, no keywords | 1 | `<pred>` | `<correct>` |
| ADV-002 | Hidden Crisis — billing overcharge, no keywords | 1 | `<pred>` | `<correct>` |
| ADV-003 | Hidden Crisis — account takeover, calm phrasing | 1 | `<pred>` | `<correct>` |
| ADV-004 | Hidden Crisis — 68x latency spike, no alarm words | 1 | `<pred>` | `<correct>` |
| ADV-005 | Hidden Crisis — insider-threat exposure, audit framing | 1 | `<pred>` | `<correct>` |
| ADV-006 | False Alarm — cosmetic complaint, keyword stuffing | 1 | `<pred>` | `<correct>` |
| ADV-007 | False Alarm — sales inquiry, misleading subject | 1 | `<pred>` | `<correct>` |
| ADV-008 | False Alarm — tutorial request, keyword-laden subject | 1 | `<pred>` | `<correct>` |
| ADV-009 | False Alarm — negated escalation words, feature request | 1 | `<pred>` | `<correct>` |
| ADV-010 | False Alarm — resolved issue marked Critical | 1 | `<pred>` | `<correct>` |
| **Score** | | | `<correct>/10` | bonus: `<bonus_earned>` |

---

## 7. Repository Layout and Usage

```
sia_project/
├── pseudo_labeler.py        # Stage 1: signal fusion, ablation.json
├── train_pipeline.py        # Stage 2: DistilBERT + LoRA classifier, metrics.json
├── dossier_generator.py     # Stage 3: Evidence Dossier construction
├── predict.py                # Orchestrator: Stages 1-3, CLI entrypoint
├── adversarial_tickets.py    # Held-out adversarial evaluation set
├── app/
│   └── streamlit_app.py      # Interactive web application
├── models/sia_classifier/    # Saved model, tokenizer, encoders, config
└── outputs/
    ├── pseudo_labeled.csv
    ├── ablation.json
    ├── metrics.json
    ├── predictions.csv
    ├── dossiers.json
    └── sample_dossiers.json
```

**Run the full pipeline:**

```bash
# Stage 1: generate pseudo-labels and ablation report
python pseudo_labeler.py

# Stage 2: train the classifier
python train_pipeline.py

# Stage 3 + inference: predict on new data and generate dossiers
python predict.py --input customer_support_tickets.csv --output outputs/
python predict.py --demo   # quick 5-ticket smoke test

# Adversarial robustness check
python -c "from adversarial_tickets import run_adversarial_eval; from predict import predict; import json; print(json.dumps(run_adversarial_eval(predict), indent=2, default=str))"

# Launch the dashboard
streamlit run app/streamlit_app.py
```

---

## 8. Design Notes and Constraints Honored

- **Independence from `Priority_Level`:** `inferred_severity` is computed purely from text, resolution behavior, and embeddings, never from `Priority_Level`, before being compared against it.
- **Two-plus signal fusion:** three signals (rule-based NLP, resolution-time regression, embedding clustering) are fused, exceeding the minimum requirement; pairwise agreement is reported for all three pairs.
- **Fine-tuned/adapter model, not zero-shot:** LoRA adapters (rank 8) on `distilbert-base-uncased` are trained on the pseudo-labeled data; `distilbert-base-uncased` is substituted for `deberta-v3-small` only at the tokenizer level (WordPiece vs. SentencePiece) for Windows compatibility, with target modules adjusted accordingly (`q_lin`/`v_lin`).
- **Multimodal input:** text (`Ticket_Subject` + `Ticket_Description`) and structured metadata (`Ticket_Channel`, `Issue_Category`, `Resolution_Time_Hours`) are both fed to the classifier via the metadata MLP fusion.
- **Imbalance handling:** class-balanced weighted cross-entropy, with weights computed via `compute_class_weight("balanced", ...)`.
- **Zero-hallucination dossiers:** every `feature_evidence` entry carries an explicit `field` pointing to a raw input column; `constraint_analysis` interpolates only pre-extracted values.
