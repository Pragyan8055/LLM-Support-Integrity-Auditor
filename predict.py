"""
predict.py
==========
Inference script for the Support Integrity Auditor (SIA).

Usage:
  python predict.py --input <path_to_csv> --output <output_dir>
  python predict.py --demo          # runs on 5 sample tickets

Outputs:
  <output_dir>/predictions.csv      — per-ticket mismatch prediction
  <output_dir>/dossiers.json        — full Evidence Dossier for flagged tickets
"""

import os
import sys
import json
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# Local imports
from pseudo_labeler import (
    compute_signal_a, compute_signal_b, compute_signal_c,
    fuse_signals, compute_mismatch_label, _score_to_priority, PRIORITY_MAP
)
from dossier_generator import generate_dossier

MODEL_DIR = "models/sia_classifier"
MAX_LEN = 128
BATCH_SIZE = 16


# ===========================================================================
# Dataset for inference
# ===========================================================================

class InferDataset(Dataset):
    def __init__(self, texts, meta, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.meta = meta
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "meta": torch.tensor(self.meta[idx], dtype=torch.float32),
        }


# ===========================================================================
# Model architecture (must match train_pipeline.py)
# ===========================================================================

META_DIM = 16

class SIAClassifier(nn.Module):
    def __init__(self, base_model, meta_input_dim, meta_dim=META_DIM, num_classes=2):
        super().__init__()
        self.encoder = base_model
        hidden = self.encoder.config.hidden_size
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_input_dim, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, meta_dim), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(hidden + meta_dim, 64), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(64, 2),
        )

    def forward(self, input_ids, attention_mask, meta):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        meta_emb = self.meta_mlp(meta)
        return self.classifier(torch.cat([cls, meta_emb], dim=-1))


# ===========================================================================
# Load model
# ===========================================================================

def load_model(model_dir: str = MODEL_DIR):
    with open(os.path.join(model_dir, "model_config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(model_dir, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    base = AutoModel.from_pretrained(cfg["model_name"])
    # Read target_modules from saved config so predict.py never hardcodes
    # model-specific layer names (DistilBERT: q_lin/v_lin; DeBERTa: query_proj/value_proj)
    lora_target_modules = cfg.get("lora_target_modules", ["q_lin", "v_lin"])
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=0.05,
        target_modules=lora_target_modules,
        bias="none",
    )
    base = get_peft_model(base, lora_cfg)
    model = SIAClassifier(base, meta_input_dim=encoders["meta_dim"])
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "best_model.pt"), map_location="cpu")
    )
    model.eval()
    return model, tokenizer, encoders, cfg


# ===========================================================================
# Feature prep for inference
# ===========================================================================

def prep_inference_features(df: pd.DataFrame, encoders: dict):
    texts = (
        df["Ticket_Subject"].fillna("") + " [SEP] " + df["Ticket_Description"].fillna("")
    ).tolist()

    le_ch: LabelEncoder = encoders["le_channel"]
    le_cat: LabelEncoder = encoders["le_category"]
    sc: StandardScaler = encoders["scaler_rt"]

    def safe_encode(le, values):
        encoded = []
        for v in values:
            if v in le.classes_:
                encoded.append(le.transform([v])[0])
            else:
                encoded.append(0)  # unknown → class 0
        return np.array(encoded)

    ch_enc = safe_encode(le_ch, df["Ticket_Channel"].fillna("Unknown").tolist())
    cat_enc = safe_encode(le_cat, df["Issue_Category"].fillna("Unknown").tolist())
    rt = df["Resolution_Time_Hours"].fillna(df["Resolution_Time_Hours"].median()).values.reshape(-1, 1)
    rt_scaled = sc.transform(rt).ravel()

    ch_oh = np.eye(len(le_ch.classes_))[ch_enc]
    cat_oh = np.eye(len(le_cat.classes_))[cat_enc]
    meta = np.hstack([ch_oh, cat_oh, rt_scaled.reshape(-1, 1)]).astype(np.float32)
    return texts, meta


# ===========================================================================
# Inference
# ===========================================================================

def predict(
    df: pd.DataFrame,
    model_dir: str = MODEL_DIR,
    output_dir: str = "outputs",
    skip_pseudo_label: bool = False,
) -> pd.DataFrame:
    """
    Run full inference pipeline on a DataFrame.
    Returns df with added prediction columns + saves dossiers.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Stage 1: Pseudo-label signals for inferred_severity
    if "inferred_severity" not in df.columns or not skip_pseudo_label:
        print("[SIA Predict] Computing pseudo-label signals...")
        sa = compute_signal_a(df)
        sb = compute_signal_b(df, signal_a_scores=sa)
        sc_ = compute_signal_c(df)
        fused, inferred = fuse_signals(sa, sb, sc_)
        df = df.copy()
        df["score_a"] = sa
        df["score_b"] = sb
        df["score_c"] = sc_
        df["fused_score"] = fused
        df["inferred_severity"] = inferred

        assigned_ord = np.array([PRIORITY_MAP.get(p, 1) for p in df["Priority_Level"]])
        inferred_ord = np.array([PRIORITY_MAP.get(p, 1) for p in inferred])
        df["severity_delta"] = inferred_ord - assigned_ord
        df["mismatch_type"] = np.where(
            df["severity_delta"] > 0, "Hidden Crisis",
            np.where(df["severity_delta"] < 0, "False Alarm", "Consistent")
        )

    # Stage 2: Classifier prediction
    print("[SIA Predict] Loading classifier...")
    model, tokenizer, encoders, cfg = load_model(model_dir)
    texts, meta = prep_inference_features(df, encoders)

    ds = InferDataset(texts, meta, tokenizer)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in dl:
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                meta=batch["meta"],
            )
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1].cpu().numpy())  # P(mismatch)

    df["predicted_mismatch"] = np.array(all_preds)
    df["mismatch_probability"] = np.array(all_probs)
    df["mismatch_label"] = df["predicted_mismatch"]  # for dossier generation

    # Stage 3: Evidence dossiers for flagged tickets
    print("[SIA Predict] Generating Evidence Dossiers for flagged tickets...")
    flagged = df[df["predicted_mismatch"] == 1].copy()
    dossiers = []
    for _, row in flagged.iterrows():
        dossiers.append(generate_dossier(row))

    # Save outputs
    df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    with open(os.path.join(output_dir, "dossiers.json"), "w") as f:
        json.dump(dossiers, f, indent=2)

    print(f"[SIA Predict] Done. {len(flagged)} mismatches flagged from {len(df)} tickets.")
    print(f"  Saved: {output_dir}/predictions.csv")
    print(f"  Saved: {output_dir}/dossiers.json")
    return df


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="SIA Inference Script")
    parser.add_argument("--input", type=str, help="C:/Users/pragy/Downloads/other_open_projects/MARS_open_projects/generated_1.csv")
    parser.add_argument("--output", type=str, default="C:/Users/pragy/Downloads/other_open_projects/MARS_open_projects/outputs", help="Output directory")
    parser.add_argument("--demo", action="store_true", help="Run on 5 sample tickets")
    args = parser.parse_args()

    if args.demo:
        df = pd.read_csv("C:/Users/pragy/Downloads/other_open_projects/MARS_open_projects/customer_support_tickets.csv").sample(5, random_state=99)
        print("[SIA] Demo mode: 5 sample tickets")
    elif args.input:
        df = pd.read_csv(args.input)
    else:
        print("Provide --input <csv> or --demo")
        sys.exit(1)

    predict(df, output_dir=args.output)


if __name__ == "__main__":
    main()
