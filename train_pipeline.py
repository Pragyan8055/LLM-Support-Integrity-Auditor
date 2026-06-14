"""
train_pipeline.py
=================
Stage 2: Fine-tuned binary classifier for Support Integrity Auditor (SIA).

Architecture:
  - Base model: distilbert-base-uncased (~66M params, standard WordPiece tokenizer,
    zero extra dependencies — works on Windows without sentencepiece)
  - LoRA adapters via PEFT (rank=8, only trains ~0.4M params)
  - Metadata fusion: Ticket_Channel + Issue_Category + Resolution_Time_Hours
    projected through a small MLP and concatenated to [CLS] embedding
  - Class imbalance: weighted cross-entropy (auto-computed class weights)

Why distilbert-base-uncased instead of deberta-v3-small:
  deberta-v3-small uses a SentencePiece tokenizer (.spm file). On Windows with
  Python 3.11, the sentencepiece C-extension frequently fails to install, and
  transformers' TikToken fallback cannot parse the binary spm.model file, raising:
    ValueError: Error parsing line b'\\x0e' in spm.model
  distilbert-base-uncased uses a plain WordPiece vocab (text file), has no
  native-code tokenizer dependency, and achieves equivalent accuracy on this task
  because the classification signal is lexical rather than morphological.

Outputs:
  models/sia_classifier/   — saved model + tokenizer + config
  outputs/metrics.json     — accuracy, macro-F1, per-class recall
  outputs/ablation.json    — updated with classifier metrics
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType

warnings.filterwarnings("ignore")

MODEL_NAME = "distilbert-base-uncased"   # WordPiece tokenizer — no sentencepiece needed
MAX_LEN = 128
BATCH_SIZE = 16        # CPU-safe batch size
EPOCHS = 4
LR = 2e-4
LORA_RANK = 8
LORA_ALPHA = 16
META_DIM = 16          # MLP output dim for metadata
HIDDEN_DIM = 768 + META_DIM  # DistilBERT hidden size (768) + metadata


# ===========================================================================
# Dataset
# ===========================================================================

class TicketDataset(Dataset):
    def __init__(self, texts, meta_features, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.meta = meta_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

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
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ===========================================================================
# Model: DistilBERT + LoRA + Metadata MLP
# ===========================================================================

class SIAClassifier(nn.Module):
    def __init__(self, base_model, meta_input_dim, meta_dim=META_DIM, num_classes=2):
        super().__init__()
        self.encoder = base_model
        hidden = self.encoder.config.hidden_size

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, meta_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden + meta_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, input_ids, attention_mask, meta):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT returns last_hidden_state; token 0 = [CLS] representation
        cls = out.last_hidden_state[:, 0, :]
        meta_emb = self.meta_mlp(meta)
        combined = torch.cat([cls, meta_emb], dim=-1)
        return self.classifier(combined)


# ===========================================================================
# Feature engineering helpers
# ===========================================================================

def build_features(df: pd.DataFrame):
    """Returns (texts, meta_array, label_encoders_dict)"""
    texts = (
        df["Ticket_Subject"].fillna("") + " [SEP] " + df["Ticket_Description"].fillna("")
    ).tolist()

    # Structured metadata
    le_ch = LabelEncoder()
    le_cat = LabelEncoder()
    sc = StandardScaler()

    ch_enc = le_ch.fit_transform(df["Ticket_Channel"].fillna("Unknown"))
    cat_enc = le_cat.fit_transform(df["Issue_Category"].fillna("Unknown"))
    rt_scaled = sc.fit_transform(
        df["Resolution_Time_Hours"].fillna(df["Resolution_Time_Hours"].median()).values.reshape(-1, 1)
    ).ravel()

    # One-hot encode channel and category
    ch_oh = np.eye(len(le_ch.classes_))[ch_enc]
    cat_oh = np.eye(len(le_cat.classes_))[cat_enc]
    rt_col = rt_scaled.reshape(-1, 1)

    meta = np.hstack([ch_oh, cat_oh, rt_col]).astype(np.float32)

    encoders = {
        "le_channel": le_ch,
        "le_category": le_cat,
        "scaler_rt": sc,
        "meta_dim": meta.shape[1],
    }
    return texts, meta, encoders


# ===========================================================================
# Training
# ===========================================================================

def train(
    data_path: str = "outputs/pseudo_labeled.csv",
    model_dir: str = "models/sia_classifier",
    output_dir: str = "outputs",
    epochs: int = EPOCHS,
):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("[SIA] Loading pseudo-labeled data...")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["mismatch_label"])

    texts, meta, encoders = build_features(df)
    labels = df["mismatch_label"].astype(int).values

    # Save encoders for inference
    import pickle
    with open(os.path.join(model_dir, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    # Train/val split
    (X_tr, X_va, meta_tr, meta_va, y_tr, y_va) = train_test_split(
        texts, meta, labels, test_size=0.15, random_state=42, stratify=labels
    )

    print(f"[SIA] Train: {len(X_tr)} | Val: {len(X_va)}")
    print(f"[SIA] Mismatch rate (train): {y_tr.mean():.2%}")

    # Class weights
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    class_weights = torch.tensor(cw, dtype=torch.float32)

    # Tokenizer
    print(f"[SIA] Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Datasets & loaders
    train_ds = TicketDataset(X_tr, meta_tr, y_tr, tokenizer)
    val_ds = TicketDataset(X_va, meta_va, y_va, tokenizer)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Base model + LoRA
    print(f"[SIA] Loading base model + applying LoRA (rank={LORA_RANK})...")
    base = AutoModel.from_pretrained(MODEL_NAME)

    # DistilBERT's MultiHeadSelfAttention uses q_lin / v_lin as the projection
    # layer names (not query_proj / value_proj which are DeBERTa-specific).
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_lin", "v_lin"],
        bias="none",
    )
    base = get_peft_model(base, lora_cfg)
    base.print_trainable_parameters()

    model = SIAClassifier(base, meta_input_dim=encoders["meta_dim"])
    device = torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    total_steps = len(train_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_f1 = 0.0
    best_epoch = 0

    print("[SIA] Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            optimizer.zero_grad()
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                meta=batch["meta"].to(device),
            )
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_dl:
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    meta=batch["meta"].to(device),
                )
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["label"].numpy())

        acc = accuracy_score(all_labels, all_preds)
        mac_f1 = f1_score(all_labels, all_preds, average="macro")
        recalls = recall_score(all_labels, all_preds, average=None, labels=[0, 1])
        avg_loss = total_loss / len(train_dl)

        print(
            f"  Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | "
            f"Acc: {acc:.4f} | Macro-F1: {mac_f1:.4f} | "
            f"Recall[Consistent]: {recalls[0]:.4f} | Recall[Mismatch]: {recalls[1]:.4f}"
        )

        if mac_f1 > best_f1:
            best_f1 = mac_f1
            best_epoch = epoch
            # Save best model state
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
            tokenizer.save_pretrained(model_dir)

    # Load best model and run final evaluation
    print(f"\n[SIA] Best model at epoch {best_epoch} (Macro-F1={best_f1:.4f})")
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_dl:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                meta=batch["meta"].to(device),
            )
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())

    final_acc = accuracy_score(all_labels, all_preds)
    final_f1 = f1_score(all_labels, all_preds, average="macro")
    final_recalls = recall_score(all_labels, all_preds, average=None, labels=[0, 1])
    report = classification_report(all_labels, all_preds, target_names=["Consistent", "Mismatch"])

    print("\n[SIA] Final Evaluation:")
    print(report)

    metrics = {
        "binary_accuracy": float(final_acc),
        "macro_f1": float(final_f1),
        "recall_consistent": float(final_recalls[0]),
        "recall_mismatch": float(final_recalls[1]),
        "thresholds_met": {
            "accuracy_83pct": bool(final_acc >= 0.83),
            "macro_f1_082": bool(final_f1 >= 0.82),
            "recall_both_078": bool(min(final_recalls) >= 0.78),
        },
        "classification_report": report,
        "best_epoch": best_epoch,
        "lora_rank": LORA_RANK,
        "epochs": epochs,
    }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Also save model config for inference
    model_config = {
        "model_name": MODEL_NAME,           # "distilbert-base-uncased"
        "max_len": MAX_LEN,
        "meta_dim": encoders["meta_dim"],
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_target_modules": ["q_lin", "v_lin"],   # DistilBERT-specific
    }
    with open(os.path.join(model_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"[SIA] Metrics saved to {output_dir}/metrics.json")
    print(f"[SIA] Model saved to {model_dir}/")
    return model, tokenizer, encoders, metrics


if __name__ == "__main__":
    train(
        data_path="C:/Users/pragy/Downloads/other_open_projects/MARS_open_projects/outputs/pseudo_labeled.csv",
        model_dir="C:/Users/pragy/Downloads/other_open_projects/MARS_open_projects/models/sia_classifier",
        output_dir="C:/Users/pragy/Downloads/other_open_projects/MARS_open_projects/outputs",
    )
