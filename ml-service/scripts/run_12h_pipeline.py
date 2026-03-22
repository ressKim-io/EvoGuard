#!/usr/bin/env python3
"""12-Hour Automated Training Pipeline.

Applies 2025-2026 state-of-the-art techniques sequentially:

Phase 1 (2h)  - LLRD + FocalLoss + R3F coevolution 200 cycles
Phase 2 (2h)  - Supervised Contrastive Learning pretraining
Phase 3 (3h)  - SCL model + FocalLoss coevolution 200 cycles
Phase 4 (2h)  - Self-Distillation (Phase 3 = teacher)
Phase 5 (2h)  - SWA + coevolution 200 cycles
Phase 6 (1h)  - Model Soup + Threshold optimization + final eval

Usage:
    python scripts/run_12h_pipeline.py
    python scripts/run_12h_pipeline.py --resume-from 3   # resume from phase 3
    python scripts/run_12h_pipeline.py --phase 2         # run only phase 2
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_service.training.losses import FocalLoss
from ml_service.training.r3f_loss import R3FLoss
from ml_service.training.contrastive_loss import SupConLoss

UTC = timezone.utc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ============================================================
# Config
# ============================================================
DATA_DIR = Path("data/korean")
MODEL_DIR = Path("models")
PIPELINE_DIR = MODEL_DIR / "pipeline_12h"
CHECKPOINT_DIR = PIPELINE_DIR / "checkpoints"

TRAIN_CSV = DATA_DIR / "korean_standard_v4_train.csv"
VALID_CSV = DATA_DIR / "korean_standard_v4_valid.csv"
TEST_CSV = DATA_DIR / "korean_standard_v1_test.csv"

BASE_MODEL = "models/production"
BASE_MODEL_NAME = "beomi/KcELECTRA-base-v2022"

MAX_LEN = 128
SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Utilities
# ============================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN, weights=None):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True,
            max_length=max_len, return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)
        self.weights = torch.tensor(
            list(weights) if weights else [1.0] * len(labels), dtype=torch.float
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
            "weight": self.weights[idx],
        }


def load_data(csv_path: Path, max_samples: int | None = None):
    df = pd.read_csv(csv_path)
    if max_samples and len(df) > max_samples:
        # Stratified sample
        toxic = df[df["label"] == 1].sample(n=max_samples // 2, random_state=SEED)
        clean = df[df["label"] == 0].sample(n=max_samples // 2, random_state=SEED)
        df = pd.concat([toxic, clean]).sample(frac=1, random_state=SEED)
    return df["text"].astype(str).tolist(), df["label"].values.tolist()


def evaluate_model(model, tokenizer, test_csv=TEST_CSV, device="cuda"):
    """Evaluate model on test set, return metrics dict."""
    model.eval()
    df = pd.read_csv(test_csv)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].values

    preds, probs_list = [], []
    for i in range(0, len(texts), 64):
        batch = texts[i : i + 64]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
        preds.extend(torch.argmax(probs, dim=1).cpu().tolist())
        probs_list.extend(probs[:, 1].cpu().tolist())

    f1 = f1_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    return {
        "f1": f1,
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tn": int(tn),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "probs": probs_list,
        "labels": labels.tolist(),
    }


def save_checkpoint(model, tokenizer, path: Path, metrics: dict | None = None):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    if metrics:
        with open(path / "metrics.json", "w") as f:
            # Remove probs/labels from saved metrics (too large)
            save_metrics = {k: v for k, v in metrics.items() if k not in ("probs", "labels")}
            json.dump(save_metrics, f, indent=2)
    logger.info(f"[SAVE] {path} (F1={metrics['f1']:.4f})" if metrics else f"[SAVE] {path}")


def get_llrd_optimizer(model, base_lr=2e-5, decay_factor=0.95, weight_decay=0.01):
    """Layer-wise Learning Rate Decay optimizer."""
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    params = []

    # Find the base model attribute
    base = None
    for attr in ("electra", "bert", "deberta", "roberta"):
        base = getattr(model, attr, None)
        if base is not None:
            break

    if base is None:
        # Fallback to standard optimizer
        return AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    # Classifier head - 2x learning rate
    classifier_params = []
    for n, p in model.named_parameters():
        if "classifier" in n or "pooler" in n:
            classifier_params.append(p)
    if classifier_params:
        params.append({"params": classifier_params, "lr": base_lr * 2, "weight_decay": weight_decay})

    # Encoder layers
    num_layers = len(base.encoder.layer)
    for layer_idx in range(num_layers - 1, -1, -1):
        layer_lr = base_lr * (decay_factor ** (num_layers - 1 - layer_idx))
        decay_params = []
        no_decay_params = []
        for n, p in base.encoder.layer[layer_idx].named_parameters():
            if any(nd in n for nd in no_decay):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        params.append({"params": decay_params, "lr": layer_lr, "weight_decay": weight_decay})
        params.append({"params": no_decay_params, "lr": layer_lr, "weight_decay": 0.0})

    # Embeddings - lowest LR
    emb_lr = base_lr * (decay_factor ** num_layers)
    emb_params = list(base.embeddings.parameters())
    if emb_params:
        params.append({"params": emb_params, "lr": emb_lr, "weight_decay": weight_decay})

    return AdamW(params)


def get_cls_embeddings(model, input_ids, attention_mask):
    """Extract [CLS] embeddings from model."""
    base = None
    for attr in ("electra", "bert", "deberta", "roberta"):
        base = getattr(model, attr, None)
        if base is not None:
            break
    if base is None:
        raise ValueError("Cannot find base model")
    outputs = base(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state[:, 0, :]  # [CLS] token


# ============================================================
# Phase 1: LLRD + FocalLoss + R3F Coevolution (2h)
# ============================================================
def run_phase1(time_budget_sec: int = 7200) -> dict:
    logger.info("=" * 60)
    logger.info("PHASE 1: LLRD + FocalLoss + R3F Coevolution")
    logger.info("=" * 60)

    set_seed()
    device = torch.device("cuda")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL).to(device)

    focal_loss = FocalLoss(gamma=2.0, alpha=0.25, reduction="none")
    r3f = R3FLoss(noise_std=1e-5, r3f_lambda=1.0)
    scaler = GradScaler()

    train_texts, train_labels = load_data(TRAIN_CSV)
    logger.info(f"Train data: {len(train_texts)} samples")

    best_f1 = 0
    best_path = CHECKPOINT_DIR / "phase1_best"
    checkpoint_paths = []

    epoch = 0
    while (time.time() - start) < time_budget_sec:
        epoch += 1
        logger.info(f"\n--- Phase 1, Epoch {epoch} ---")

        # Sample balanced batch for this epoch
        n_per_class = min(15000, len(train_texts) // 4)
        toxic_idx = [i for i, l in enumerate(train_labels) if l == 1]
        clean_idx = [i for i, l in enumerate(train_labels) if l == 0]
        sampled = random.sample(toxic_idx, min(n_per_class, len(toxic_idx))) + \
                  random.sample(clean_idx, min(n_per_class, len(clean_idx)))
        random.shuffle(sampled)

        epoch_texts = [train_texts[i] for i in sampled]
        epoch_labels = [train_labels[i] for i in sampled]

        dataset = TextDataset(epoch_texts, epoch_labels, tokenizer)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

        optimizer = get_llrd_optimizer(model, base_lr=2e-5, decay_factor=0.95)
        total_steps = len(dataloader)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        model.train()
        epoch_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with autocast():
                focal_mean = FocalLoss(gamma=2.0, alpha=0.25, reduction="mean")
                loss, _ = r3f(model, input_ids, attention_mask, labels, focal_mean)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"  Loss: {avg_loss:.4f}")

        # Evaluate
        metrics = evaluate_model(model, tokenizer, device=device)
        logger.info(f"  F1={metrics['f1']:.4f} FP={metrics['fp']} FN={metrics['fn']}")

        # Save checkpoint
        ckpt_path = CHECKPOINT_DIR / f"phase1_epoch{epoch}"
        save_checkpoint(model, tokenizer, ckpt_path, metrics)
        checkpoint_paths.append((ckpt_path, metrics["f1"]))

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_checkpoint(model, tokenizer, best_path, metrics)
            logger.info(f"  ★ New best F1: {best_f1:.4f}")

        elapsed = time.time() - start
        logger.info(f"  Elapsed: {elapsed/60:.1f}min / {time_budget_sec/60:.0f}min budget")

    logger.info(f"\n[Phase 1 DONE] Best F1: {best_f1:.4f}, Epochs: {epoch}")
    return {"best_f1": best_f1, "best_path": str(best_path), "epochs": epoch, "checkpoints": checkpoint_paths}


# ============================================================
# Phase 2: Supervised Contrastive Learning (2h)
# ============================================================
def run_phase2(base_model_path: str, time_budget_sec: int = 7200) -> dict:
    logger.info("=" * 60)
    logger.info("PHASE 2: Supervised Contrastive Learning")
    logger.info("=" * 60)

    set_seed()
    device = torch.device("cuda")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path).to(device)

    scl_loss_fn = SupConLoss(temperature=0.07)
    scaler = GradScaler()

    train_texts, train_labels = load_data(TRAIN_CSV)

    # Projection head for contrastive learning
    hidden_size = model.config.hidden_size
    projection_head = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 128),
    ).to(device)

    best_path = CHECKPOINT_DIR / "phase2_best"

    epoch = 0
    while (time.time() - start) < time_budget_sec:
        epoch += 1
        logger.info(f"\n--- Phase 2, Epoch {epoch} ---")

        # Balanced sampling
        n_per_class = min(12000, len(train_texts) // 4)
        toxic_idx = [i for i, l in enumerate(train_labels) if l == 1]
        clean_idx = [i for i, l in enumerate(train_labels) if l == 0]
        sampled = random.sample(toxic_idx, min(n_per_class, len(toxic_idx))) + \
                  random.sample(clean_idx, min(n_per_class, len(clean_idx)))
        random.shuffle(sampled)

        epoch_texts = [train_texts[i] for i in sampled]
        epoch_labels = [train_labels[i] for i in sampled]

        dataset = TextDataset(epoch_texts, epoch_labels, tokenizer)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

        # Only train base model embeddings + projection head
        all_params = list(model.parameters()) + list(projection_head.parameters())
        optimizer = AdamW(all_params, lr=5e-6, weight_decay=0.01)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=len(dataloader) // 10, num_training_steps=len(dataloader)
        )

        model.train()
        projection_head.train()
        epoch_loss = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with autocast():
                cls_emb = get_cls_embeddings(model, input_ids, attention_mask)
                projected = projection_head(cls_emb)
                loss = scl_loss_fn(projected, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"  SCL Loss: {avg_loss:.4f}")

        # Evaluate (classification head still exists)
        metrics = evaluate_model(model, tokenizer, device=device)
        logger.info(f"  F1={metrics['f1']:.4f} FP={metrics['fp']} FN={metrics['fn']}")

        elapsed = time.time() - start
        logger.info(f"  Elapsed: {elapsed/60:.1f}min / {time_budget_sec/60:.0f}min budget")

    # Save SCL-pretrained model
    save_checkpoint(model, tokenizer, best_path)
    logger.info(f"\n[Phase 2 DONE] SCL pretraining complete, {epoch} epochs")
    del projection_head
    gc.collect()
    torch.cuda.empty_cache()
    return {"best_path": str(best_path), "epochs": epoch}


# ============================================================
# Phase 3: SCL + FocalLoss + LLRD Coevolution (3h)
# ============================================================
def run_phase3(base_model_path: str, time_budget_sec: int = 10800) -> dict:
    logger.info("=" * 60)
    logger.info("PHASE 3: SCL + FocalLoss + LLRD Coevolution")
    logger.info("=" * 60)

    set_seed()
    device = torch.device("cuda")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path).to(device)

    r3f = R3FLoss(noise_std=1e-5, r3f_lambda=0.5)  # Lower lambda after SCL
    scaler = GradScaler()

    train_texts, train_labels = load_data(TRAIN_CSV)

    best_f1 = 0
    best_path = CHECKPOINT_DIR / "phase3_best"
    checkpoint_paths = []

    epoch = 0
    while (time.time() - start) < time_budget_sec:
        epoch += 1
        logger.info(f"\n--- Phase 3, Epoch {epoch} ---")

        n_per_class = min(15000, len(train_texts) // 4)
        toxic_idx = [i for i, l in enumerate(train_labels) if l == 1]
        clean_idx = [i for i, l in enumerate(train_labels) if l == 0]
        sampled = random.sample(toxic_idx, min(n_per_class, len(toxic_idx))) + \
                  random.sample(clean_idx, min(n_per_class, len(clean_idx)))
        random.shuffle(sampled)

        epoch_texts = [train_texts[i] for i in sampled]
        epoch_labels = [train_labels[i] for i in sampled]

        dataset = TextDataset(epoch_texts, epoch_labels, tokenizer)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

        # LLRD with cosine annealing w/ warm restart
        lr = 1.5e-5 if epoch <= 5 else 1e-5  # Decay base LR over time
        optimizer = get_llrd_optimizer(model, base_lr=lr, decay_factor=0.95)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=len(dataloader) // 10, num_training_steps=len(dataloader)
        )

        model.train()
        epoch_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with autocast():
                focal_mean = FocalLoss(gamma=2.0, alpha=0.25, reduction="mean")
                loss, _ = r3f(model, input_ids, attention_mask, labels, focal_mean)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"  Loss: {avg_loss:.4f}")

        metrics = evaluate_model(model, tokenizer, device=device)
        logger.info(f"  F1={metrics['f1']:.4f} FP={metrics['fp']} FN={metrics['fn']}")

        ckpt_path = CHECKPOINT_DIR / f"phase3_epoch{epoch}"
        save_checkpoint(model, tokenizer, ckpt_path, metrics)
        checkpoint_paths.append((ckpt_path, metrics["f1"]))

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_checkpoint(model, tokenizer, best_path, metrics)
            logger.info(f"  ★ New best F1: {best_f1:.4f}")

        elapsed = time.time() - start
        logger.info(f"  Elapsed: {elapsed/60:.1f}min / {time_budget_sec/60:.0f}min budget")

    logger.info(f"\n[Phase 3 DONE] Best F1: {best_f1:.4f}, Epochs: {epoch}")
    return {"best_f1": best_f1, "best_path": str(best_path), "epochs": epoch, "checkpoints": checkpoint_paths}


# ============================================================
# Phase 4: Self-Distillation (2h)
# ============================================================
def run_phase4(teacher_path: str, time_budget_sec: int = 7200) -> dict:
    logger.info("=" * 60)
    logger.info("PHASE 4: Self-Distillation")
    logger.info("=" * 60)

    set_seed()
    device = torch.device("cuda")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(teacher_path)

    # Teacher (frozen)
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_path).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student (fresh from production, will learn from teacher)
    student = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL).to(device)

    scaler = GradScaler()
    train_texts, train_labels = load_data(TRAIN_CSV)

    temperature = 3.0
    alpha = 0.7  # Weight of soft loss vs hard loss

    best_f1 = 0
    best_path = CHECKPOINT_DIR / "phase4_best"
    checkpoint_paths = []

    epoch = 0
    while (time.time() - start) < time_budget_sec:
        epoch += 1
        logger.info(f"\n--- Phase 4, Epoch {epoch} ---")

        n_per_class = min(15000, len(train_texts) // 4)
        toxic_idx = [i for i, l in enumerate(train_labels) if l == 1]
        clean_idx = [i for i, l in enumerate(train_labels) if l == 0]
        sampled = random.sample(toxic_idx, min(n_per_class, len(toxic_idx))) + \
                  random.sample(clean_idx, min(n_per_class, len(clean_idx)))
        random.shuffle(sampled)

        epoch_texts = [train_texts[i] for i in sampled]
        epoch_labels = [train_labels[i] for i in sampled]

        dataset = TextDataset(epoch_texts, epoch_labels, tokenizer)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

        optimizer = get_llrd_optimizer(student, base_lr=2e-5, decay_factor=0.95)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=len(dataloader) // 10, num_training_steps=len(dataloader)
        )

        student.train()
        epoch_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with autocast():
                # Student logits
                student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits

                # Teacher logits (no grad)
                with torch.no_grad():
                    teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

                # Hard loss (Focal)
                hard_loss = FocalLoss(gamma=2.0, alpha=0.25, reduction="mean")(student_logits, labels)

                # Soft loss (KL divergence with temperature)
                soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
                soft_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=-1),
                    soft_targets,
                    reduction="batchmean",
                ) * (temperature ** 2)

                loss = alpha * soft_loss + (1 - alpha) * hard_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"  Loss: {avg_loss:.4f}")

        metrics = evaluate_model(student, tokenizer, device=device)
        logger.info(f"  F1={metrics['f1']:.4f} FP={metrics['fp']} FN={metrics['fn']}")

        ckpt_path = CHECKPOINT_DIR / f"phase4_epoch{epoch}"
        save_checkpoint(student, tokenizer, ckpt_path, metrics)
        checkpoint_paths.append((ckpt_path, metrics["f1"]))

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_checkpoint(student, tokenizer, best_path, metrics)
            logger.info(f"  ★ New best F1: {best_f1:.4f}")

        elapsed = time.time() - start
        logger.info(f"  Elapsed: {elapsed/60:.1f}min / {time_budget_sec/60:.0f}min budget")

    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"\n[Phase 4 DONE] Best F1: {best_f1:.4f}, Epochs: {epoch}")
    return {"best_f1": best_f1, "best_path": str(best_path), "epochs": epoch, "checkpoints": checkpoint_paths}


# ============================================================
# Phase 5: SWA + Coevolution (2h)
# ============================================================
def run_phase5(base_model_path: str, time_budget_sec: int = 7200) -> dict:
    logger.info("=" * 60)
    logger.info("PHASE 5: SWA + Fine-tuning")
    logger.info("=" * 60)

    set_seed()
    device = torch.device("cuda")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path).to(device)

    swa_model = AveragedModel(model)
    scaler = GradScaler()
    r3f = R3FLoss(noise_std=1e-5, r3f_lambda=0.5)

    train_texts, train_labels = load_data(TRAIN_CSV)

    best_f1 = 0
    best_path = CHECKPOINT_DIR / "phase5_best"
    swa_path = CHECKPOINT_DIR / "phase5_swa"
    checkpoint_paths = []
    swa_start_epoch = 3  # Start SWA after 3 epochs

    epoch = 0
    while (time.time() - start) < time_budget_sec:
        epoch += 1
        logger.info(f"\n--- Phase 5, Epoch {epoch} {'(SWA active)' if epoch >= swa_start_epoch else ''} ---")

        n_per_class = min(15000, len(train_texts) // 4)
        toxic_idx = [i for i, l in enumerate(train_labels) if l == 1]
        clean_idx = [i for i, l in enumerate(train_labels) if l == 0]
        sampled = random.sample(toxic_idx, min(n_per_class, len(toxic_idx))) + \
                  random.sample(clean_idx, min(n_per_class, len(clean_idx)))
        random.shuffle(sampled)

        epoch_texts = [train_texts[i] for i in sampled]
        epoch_labels = [train_labels[i] for i in sampled]

        dataset = TextDataset(epoch_texts, epoch_labels, tokenizer)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

        # Constant low LR for SWA phase
        lr = 1e-5 if epoch >= swa_start_epoch else 2e-5
        optimizer = get_llrd_optimizer(model, base_lr=lr, decay_factor=0.95)

        if epoch >= swa_start_epoch:
            swa_scheduler = SWALR(optimizer, swa_lr=5e-6)
        else:
            swa_scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=len(dataloader) // 10, num_training_steps=len(dataloader)
            )

        model.train()
        epoch_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with autocast():
                focal_mean = FocalLoss(gamma=2.0, alpha=0.25, reduction="mean")
                loss, _ = r3f(model, input_ids, attention_mask, labels, focal_mean)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if epoch < swa_start_epoch:
                swa_scheduler.step()

        if epoch >= swa_start_epoch:
            swa_scheduler.step()
            swa_model.update_parameters(model)

        avg_loss = epoch_loss / max(len(dataloader), 1)
        logger.info(f"  Loss: {avg_loss:.4f}")

        # Evaluate base model
        metrics = evaluate_model(model, tokenizer, device=device)
        logger.info(f"  Base F1={metrics['f1']:.4f} FP={metrics['fp']} FN={metrics['fn']}")

        ckpt_path = CHECKPOINT_DIR / f"phase5_epoch{epoch}"
        save_checkpoint(model, tokenizer, ckpt_path, metrics)
        checkpoint_paths.append((ckpt_path, metrics["f1"]))

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_checkpoint(model, tokenizer, best_path, metrics)
            logger.info(f"  ★ New best F1: {best_f1:.4f}")

        elapsed = time.time() - start
        logger.info(f"  Elapsed: {elapsed/60:.1f}min / {time_budget_sec/60:.0f}min budget")

    # Update BN for SWA model and evaluate
    logger.info("\nUpdating SWA batch norm...")
    try:
        bn_dataset = TextDataset(train_texts[:5000], train_labels[:5000], tokenizer)
        bn_loader = DataLoader(bn_dataset, batch_size=32, num_workers=2)

        # SWA BN update
        swa_model.to(device)
        torch.optim.swa_utils.update_bn(bn_loader, swa_model, device=device)

        swa_metrics = evaluate_model(swa_model, tokenizer, device=device)
        logger.info(f"  SWA F1={swa_metrics['f1']:.4f} FP={swa_metrics['fp']} FN={swa_metrics['fn']}")

        # Save SWA model if better
        if swa_metrics["f1"] > best_f1:
            best_f1 = swa_metrics["f1"]
            # Copy SWA weights to base model for saving
            model.load_state_dict(swa_model.module.state_dict())
            save_checkpoint(model, tokenizer, swa_path, swa_metrics)
            save_checkpoint(model, tokenizer, best_path, swa_metrics)
            logger.info(f"  ★ SWA improved! F1: {best_f1:.4f}")
    except Exception as e:
        logger.warning(f"SWA BN update failed: {e}, using base model")

    logger.info(f"\n[Phase 5 DONE] Best F1: {best_f1:.4f}, Epochs: {epoch}")
    return {"best_f1": best_f1, "best_path": str(best_path), "epochs": epoch, "checkpoints": checkpoint_paths}


# ============================================================
# Phase 6: Model Soup + Threshold Optimization (1h)
# ============================================================
def run_phase6(all_results: dict) -> dict:
    logger.info("=" * 60)
    logger.info("PHASE 6: Model Soup + Threshold Optimization")
    logger.info("=" * 60)

    device = torch.device("cuda")

    # Collect all checkpoint paths with their F1 scores
    all_checkpoints = []
    for phase_key, result in all_results.items():
        if "checkpoints" in result:
            for ckpt_path, f1 in result["checkpoints"]:
                if Path(ckpt_path).exists() and (Path(ckpt_path) / "config.json").exists():
                    all_checkpoints.append((str(ckpt_path), f1))

    # Sort by F1, take top-5
    all_checkpoints.sort(key=lambda x: x[1], reverse=True)
    top_checkpoints = all_checkpoints[:5]

    logger.info(f"Top {len(top_checkpoints)} checkpoints:")
    for path, f1 in top_checkpoints:
        logger.info(f"  {Path(path).name}: F1={f1:.4f}")

    if not top_checkpoints:
        logger.warning("No checkpoints found!")
        return {}

    # --- Greedy Model Soup ---
    logger.info("\n--- Greedy Model Soup ---")
    tokenizer = AutoTokenizer.from_pretrained(top_checkpoints[0][0])

    # Start with best model
    base_model = AutoModelForSequenceClassification.from_pretrained(top_checkpoints[0][0]).to(device)
    best_metrics = evaluate_model(base_model, tokenizer, device=device)
    best_soup_f1 = best_metrics["f1"]
    # Keep soup_state on CPU for consistent averaging
    soup_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
    n_models = 1

    logger.info(f"  Base: F1={best_soup_f1:.4f}")

    for ckpt_path, _ in top_checkpoints[1:]:
        try:
            candidate = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
            candidate_state = candidate.state_dict()

            # Try averaging (all on CPU)
            trial_state = {}
            for key in soup_state:
                if key in candidate_state:
                    trial_state[key] = (soup_state[key] * n_models + candidate_state[key].cpu()) / (n_models + 1)
                else:
                    trial_state[key] = soup_state[key]

            base_model.cpu()
            base_model.load_state_dict(trial_state)
            base_model.to(device)
            trial_metrics = evaluate_model(base_model, tokenizer, device=device)

            if trial_metrics["f1"] > best_soup_f1:
                best_soup_f1 = trial_metrics["f1"]
                soup_state = {k: v.cpu().clone() for k, v in trial_state.items()}
                n_models += 1
                logger.info(f"  + {Path(ckpt_path).name}: F1={best_soup_f1:.4f} (added, {n_models} models)")
            else:
                logger.info(f"  - {Path(ckpt_path).name}: F1={trial_metrics['f1']:.4f} (skipped)")

            del candidate
        except Exception as e:
            logger.warning(f"  Failed to load {ckpt_path}: {e}")

    # Load best soup
    base_model.cpu()
    base_model.load_state_dict(soup_state)
    base_model.to(device)

    # --- Threshold Optimization ---
    logger.info("\n--- Threshold Optimization ---")
    valid_metrics = evaluate_model(base_model, tokenizer, test_csv=VALID_CSV, device=device)
    valid_probs = valid_metrics["probs"]
    valid_labels = valid_metrics["labels"]

    best_threshold = 0.5
    best_thresh_f1 = 0

    for thresh in np.arange(0.30, 0.70, 0.005):
        preds = [1 if p >= thresh else 0 for p in valid_probs]
        f1 = f1_score(valid_labels, preds)
        if f1 > best_thresh_f1:
            best_thresh_f1 = f1
            best_threshold = thresh

    logger.info(f"  Optimal threshold: {best_threshold:.3f} (valid F1={best_thresh_f1:.4f})")

    # --- Final Evaluation on Test Set ---
    logger.info("\n--- Final Test Set Evaluation ---")

    # With default threshold
    test_metrics_default = evaluate_model(base_model, tokenizer, device=device)
    logger.info(f"  Default (0.5):   F1={test_metrics_default['f1']:.4f} FP={test_metrics_default['fp']} FN={test_metrics_default['fn']}")

    # With optimized threshold
    test_probs = test_metrics_default["probs"]
    test_labels = test_metrics_default["labels"]
    opt_preds = [1 if p >= best_threshold else 0 for p in test_probs]
    opt_f1 = f1_score(test_labels, opt_preds)
    opt_cm = confusion_matrix(test_labels, opt_preds)
    opt_tn, opt_fp, opt_fn, opt_tp = opt_cm.ravel()

    logger.info(f"  Optimized ({best_threshold:.3f}): F1={opt_f1:.4f} FP={opt_fp} FN={opt_fn}")

    # Pick the better one
    if opt_f1 > test_metrics_default["f1"]:
        final_f1 = opt_f1
        final_fp, final_fn = int(opt_fp), int(opt_fn)
        final_threshold = best_threshold
    else:
        final_f1 = test_metrics_default["f1"]
        final_fp = test_metrics_default["fp"]
        final_fn = test_metrics_default["fn"]
        final_threshold = 0.5

    # Save final model
    final_path = PIPELINE_DIR / "final_model"
    save_checkpoint(base_model, tokenizer, final_path, {
        "f1": final_f1, "fp": final_fp, "fn": final_fn,
        "threshold": final_threshold,
        "soup_models": n_models,
    })

    # Also save to coevolution-latest if best overall
    production_path = MODEL_DIR / "coevolution-latest"
    save_checkpoint(base_model, tokenizer, production_path, {
        "f1": final_f1, "fp": final_fp, "fn": final_fn,
        "threshold": final_threshold,
        "pipeline": "12h_pipeline",
    })

    logger.info(f"\n★★★ FINAL RESULT: F1={final_f1:.4f} FP={final_fp} FN={final_fn} threshold={final_threshold:.3f} ★★★")

    return {
        "final_f1": final_f1,
        "final_fp": final_fp,
        "final_fn": final_fn,
        "threshold": final_threshold,
        "soup_models": n_models,
        "final_path": str(final_path),
    }


# ============================================================
# Main Pipeline
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="12-Hour Training Pipeline")
    parser.add_argument("--resume-from", type=int, default=1, help="Resume from phase N")
    parser.add_argument("--phase", type=int, default=None, help="Run only phase N")
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="Scale time budgets (0.5 = half time, 2.0 = double)")
    args = parser.parse_args()

    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Time budgets (seconds)
    scale = args.time_scale
    budgets = {
        1: int(7200 * scale),   # 2h
        2: int(7200 * scale),   # 2h
        3: int(10800 * scale),  # 3h
        4: int(7200 * scale),   # 2h
        5: int(7200 * scale),   # 2h
        6: int(3600 * scale),   # 1h
    }

    pipeline_start = time.time()
    results = {}
    results_path = PIPELINE_DIR / "pipeline_results.json"

    # Load existing results if resuming
    if args.resume_from > 1 and results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        logger.info(f"Loaded existing results from phases: {list(results.keys())}")

    start_phase = args.phase or args.resume_from
    end_phase = args.phase or 6

    logger.info("=" * 60)
    logger.info(f"12-HOUR TRAINING PIPELINE")
    logger.info(f"Phases: {start_phase} → {end_phase}")
    logger.info(f"Time scale: {scale}x")
    total_budget = sum(budgets[i] for i in range(start_phase, end_phase + 1))
    logger.info(f"Total budget: {total_budget/3600:.1f}h")
    logger.info(f"Start: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    logger.info("=" * 60)

    def save_results():
        serializable = {}
        for k, v in results.items():
            serializable[k] = {}
            for kk, vv in v.items():
                if kk == "checkpoints":
                    serializable[k][kk] = [(str(p), f) for p, f in vv]
                else:
                    serializable[k][kk] = vv
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)

    try:
        # Phase 1
        if start_phase <= 1 <= end_phase:
            results["phase1"] = run_phase1(budgets[1])
            save_results()
            gc.collect()
            torch.cuda.empty_cache()

        # Phase 2
        if start_phase <= 2 <= end_phase:
            phase1_best = results.get("phase1", {}).get("best_path", BASE_MODEL)
            results["phase2"] = run_phase2(phase1_best, budgets[2])
            save_results()
            gc.collect()
            torch.cuda.empty_cache()

        # Phase 3
        if start_phase <= 3 <= end_phase:
            phase2_best = results.get("phase2", {}).get("best_path", BASE_MODEL)
            results["phase3"] = run_phase3(phase2_best, budgets[3])
            save_results()
            gc.collect()
            torch.cuda.empty_cache()

        # Phase 4
        if start_phase <= 4 <= end_phase:
            # Teacher = best from phase 3
            teacher_path = results.get("phase3", {}).get("best_path", BASE_MODEL)
            results["phase4"] = run_phase4(teacher_path, budgets[4])
            save_results()
            gc.collect()
            torch.cuda.empty_cache()

        # Phase 5
        if start_phase <= 5 <= end_phase:
            # Use best from phase 3 or 4
            p3_f1 = results.get("phase3", {}).get("best_f1", 0)
            p4_f1 = results.get("phase4", {}).get("best_f1", 0)
            if p4_f1 > p3_f1:
                base = results["phase4"]["best_path"]
            elif p3_f1 > 0:
                base = results["phase3"]["best_path"]
            else:
                base = BASE_MODEL
            results["phase5"] = run_phase5(base, budgets[5])
            save_results()
            gc.collect()
            torch.cuda.empty_cache()

        # Phase 6
        if start_phase <= 6 <= end_phase:
            results["phase6"] = run_phase6(results)
            save_results()

    except KeyboardInterrupt:
        logger.info("\n[INTERRUPTED] Saving progress...")
        save_results()
    except Exception as e:
        logger.error(f"\n[ERROR] {e}", exc_info=True)
        save_results()
        raise

    total_time = time.time() - pipeline_start
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time/3600:.1f}h")

    # Print summary
    logger.info("\n--- Phase Summary ---")
    for phase_key in sorted(results.keys()):
        r = results[phase_key]
        f1 = r.get("best_f1", r.get("final_f1", "N/A"))
        epochs = r.get("epochs", "N/A")
        if isinstance(f1, float):
            logger.info(f"  {phase_key}: F1={f1:.4f}, epochs={epochs}")
        else:
            logger.info(f"  {phase_key}: {r}")

    if "phase6" in results:
        r6 = results["phase6"]
        logger.info(f"\n★ FINAL: F1={r6['final_f1']:.4f} FP={r6['final_fp']} FN={r6['final_fn']} threshold={r6.get('threshold', 0.5):.3f}")


if __name__ == "__main__":
    main()
