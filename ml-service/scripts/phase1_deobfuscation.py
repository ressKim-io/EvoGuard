#!/usr/bin/env python3
"""Phase 1: Train with deobfuscated KOTOX data.

Uses KOTOX rules to normalize obfuscated text before classification.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class KOTOXDeobfuscator:
    """Deobfuscate Korean text using KOTOX rules."""

    def __init__(self, rules_dir: Path):
        self.rules_dir = rules_dir
        self._load_rules()

    def _load_rules(self):
        """Load KOTOX deobfuscation rules."""
        # Load iconic dictionary (special characters -> Korean)
        iconic_path = self.rules_dir / "iconic_dictionary.json"
        if iconic_path.exists():
            with open(iconic_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.consonant_map = {}
                self.vowel_map = {}

                # Build reverse mapping (obfuscated -> original)
                for orig, variants in data.get("consonant_dict", {}).items():
                    for v in variants:
                        self.consonant_map[v] = orig

                for orig, variants in data.get("vowel_dict", {}).items():
                    for v in variants:
                        self.vowel_map[v] = orig
        else:
            self.consonant_map = {}
            self.vowel_map = {}

        # Load transliteration dictionary
        trans_path = self.rules_dir / "transliterational_dictionary.json"
        if trans_path.exists():
            with open(trans_path, "r", encoding="utf-8") as f:
                self.trans_map = json.load(f)
        else:
            self.trans_map = {}

        logger.info(f"Loaded {len(self.consonant_map)} consonant mappings, "
                   f"{len(self.vowel_map)} vowel mappings")

    def deobfuscate(self, text: str) -> str:
        """Apply deobfuscation rules to text."""
        result = text

        # Apply consonant mappings
        for obf, orig in self.consonant_map.items():
            result = result.replace(obf, orig)

        # Apply vowel mappings
        for obf, orig in self.vowel_map.items():
            result = result.replace(obf, orig)

        # Remove zero-width characters
        result = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', result)

        # Normalize whitespace
        result = re.sub(r'\s+', ' ', result).strip()

        return result


class ToxicDataset(Dataset):
    """Dataset for toxic text classification."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Deobfuscation Training")
    parser.add_argument("--model", default="beomi/KcELECTRA-base-v2022")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", default="models/phase1-deobfuscated")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load deobfuscator
    rules_dir = Path(__file__).parent.parent / "data" / "korean" / "KOTOX" / "rules"
    deobfuscator = KOTOXDeobfuscator(rules_dir)

    # Load data
    data_dir = Path(__file__).parent.parent / "data" / "korean" / "KOTOX" / "data" / "KOTOX_classification" / "total"

    train_df = pd.read_csv(data_dir / "train.csv")
    valid_df = pd.read_csv(data_dir / "valid.csv")

    # Shuffle
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    valid_df = valid_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Apply deobfuscation
    logger.info("Applying deobfuscation to training data...")
    train_df["text_clean"] = train_df["text"].apply(deobfuscator.deobfuscate)
    valid_df["text_clean"] = valid_df["text"].apply(deobfuscator.deobfuscate)

    logger.info(f"Train: {len(train_df)}, Valid: {len(valid_df)}")
    logger.info(f"Sample original: {train_df.iloc[0]['text'][:50]}")
    logger.info(f"Sample cleaned:  {train_df.iloc[0]['text_clean'][:50]}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)

    # Create datasets
    train_dataset = ToxicDataset(train_df["text_clean"].tolist(), train_df["label"].tolist(), tokenizer)
    valid_dataset = ToxicDataset(valid_df["text_clean"].tolist(), valid_df["label"].tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    # Training
    from ml_service.training.losses import FocalLoss

    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_f1 = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHASE 1: DEOBFUSCATION TRAINING")
    logger.info("=" * 60)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].tolist())

        from sklearn.metrics import f1_score, accuracy_score
        f1 = f1_score(all_labels, all_preds, average="weighted")
        acc = accuracy_score(all_labels, all_preds)

        logger.info(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, f1={f1:.4f}, acc={acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")
            logger.info(f"New best model saved (F1={best_f1:.4f})")

    logger.info("=" * 60)
    logger.info(f"PHASE 1 COMPLETE - Best F1: {best_f1:.4f}")
    logger.info("=" * 60)

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump({"best_f1": best_f1, "epochs": args.epochs}, f)


if __name__ == "__main__":
    main()
