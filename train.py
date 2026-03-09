"""
Training Script — MTL Model (Intent + Emotion)
================================================
For demo purposes, this script uses a small synthetic dataset
so you can verify the training loop works end-to-end before
plugging in your real labeled data.

Usage:
    python train.py --epochs 3 --batch_size 8 --output_dir ./checkpoints
"""

import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

import sys
sys.path.insert(0, os.path.dirname(__file__))

from model.mtl_model import MTLIntentEmotionModel, MTLLoss, INTENT_LABELS, EMOTION_LABELS


# ─────────────────────────────────────────────
# Synthetic Dataset
# ─────────────────────────────────────────────
# Replace with your real data. Each entry is:
#   (text, [active_intent_indices], emotion_index)

SYNTHETIC_DATA = [
    ("My card was stolen this morning, please help me cancel it immediately.",
     [0, 2], 3),    # report_theft + cancel_card | Urgent
    ("What's my current account balance?",
     [3], 0),       # check_balance | Neutral
    ("Oh sure, my money just disappeared magically. Very helpful system you have.",
     [4], 2),       # speak_to_agent | Sarcasm
    ("I need emergency cash, my wallet was taken at the airport in London.",
     [0, 1], 3),    # report_theft + request_emergency_cash | Urgent
    ("Can I update my address on file?",
     [6], 0),       # update_details | Neutral
    ("I've called three times and nobody is fixing this! I need a human NOW.",
     [4], 1),       # speak_to_agent | Frustration
    ("Just a general question about your fees.",
     [5], 0),       # general_inquiry | Neutral
    ("I'm so frustrated! The app keeps crashing and I can't check my balance.",
     [3], 1),       # check_balance | Frustration
]


class SyntheticMTLDataset(Dataset):
    def __init__(self, data, tokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, intent_indices, emotion_idx = self.data[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Multi-hot intent target (float for BCEWithLogitsLoss)
        intent_target = torch.zeros(len(INTENT_LABELS), dtype=torch.float)
        for i in intent_indices:
            intent_target[i] = 1.0

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "intent_target": intent_target,
            "emotion_target": torch.tensor(emotion_idx, dtype=torch.long),
        }


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    model = MTLIntentEmotionModel(
        encoder_name=args.encoder,
        freeze_encoder_layers=8,
    ).to(device)

    dataset = SyntheticMTLDataset(SYNTHETIC_DATA, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    loss_fn = MTLLoss(alpha_intent=0.5, alpha_emotion=0.5)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_targets = batch["intent_target"].to(device)
            emotion_targets = batch["emotion_target"].to(device)

            intent_logits, emotion_logits = model(input_ids, attention_mask)

            losses = loss_fn(intent_logits, emotion_logits, intent_targets, emotion_targets)
            losses["total"].backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += losses["total"].item()

        avg = epoch_loss / len(loader)
        print(f"Epoch {epoch}/{args.epochs} — Loss: {avg:.4f}")

    # ── Save checkpoint ──────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "mtl_model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nModel saved to: {checkpoint_path}")
    print("To load: model.load_state_dict(torch.load(checkpoint_path))")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="roberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", default="./checkpoints")
    args = parser.parse_args()
    train(args)
