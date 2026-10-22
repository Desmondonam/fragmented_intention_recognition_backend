"""
Training Script — MTL Model (Intent + Emotion) and Single-Task Baselines
=========================================================================
The same script, architecture and hyperparameters produce all three runs, so
the only thing that changes between them is which task losses are active:

    python -m training.train --tasks intent,emotion --output_dir runs/mtl
    python -m training.train --tasks intent         --output_dir runs/stl_intent
    python -m training.train --tasks emotion        --output_dir runs/stl_emotion

(scripts/run_experiments.sh runs all three with the settings reported in the README.)

Each epoch makes one full pass over every active task's training set, with
batches from the two tasks interleaved in random order. A single-task run
therefore sees exactly the same data for its task as the multi-task run.

The epoch with the best mean validation macro-F1 (over active tasks) is kept
and evaluated once on the test split; everything is written to metrics.json.
"""

import argparse
import json
import os
import random
import time
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from model.mtl_model import MTLIntentEmotionModel, MTLLoss
from training.data import TASKS, TextClassificationDataset, load_task, make_collate_fn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def logits_for(task: str, intent_logits, emotion_logits):
    return intent_logits if task == "intent" else emotion_logits


@torch.no_grad()
def evaluate(model, loader, task: str, device: str) -> Dict[str, float]:
    model.eval()
    preds, gold = [], []
    for batch in loader:
        intent_logits, emotion_logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        preds += logits_for(task, intent_logits, emotion_logits).argmax(-1).tolist()
        gold += batch["labels"].tolist()
    return {
        "accuracy": round(accuracy_score(gold, preds), 4),
        "macro_f1": round(f1_score(gold, preds, average="macro"), 4),
        "n": len(gold),
    }


def train(args):
    set_seed(args.seed)
    torch.set_num_threads(args.threads or os.cpu_count())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = [t.strip() for t in args.tasks.split(",")]
    assert all(t in TASKS for t in tasks), f"--tasks must be drawn from {TASKS}"
    print(f"Training tasks={tasks} on {device} with encoder={args.encoder}")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    collate = make_collate_fn(tokenizer, args.max_length)

    loaders = {}
    for task in tasks:
        splits = load_task(task, seed=args.seed)
        train_split = splits["train"].sample(args.max_train_per_task, args.seed)
        val_split = splits["validation"].sample(args.max_eval_per_task, args.seed)
        print(f"  {task}: train={len(train_split)} val={len(splits['validation'])} test={len(splits['test'])}")
        loaders[task] = {
            "train": DataLoader(TextClassificationDataset(train_split), batch_size=args.batch_size,
                                shuffle=True, collate_fn=collate),
            "validation": DataLoader(TextClassificationDataset(val_split),
                                     batch_size=args.eval_batch_size, collate_fn=collate),
            "test": DataLoader(TextClassificationDataset(splits["test"]),
                               batch_size=args.eval_batch_size, collate_fn=collate),
        }

    model = MTLIntentEmotionModel(encoder_name=args.encoder, freeze_encoder_layers=args.freeze_layers).to(device)
    loss_fn = MTLLoss()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)

    steps_per_epoch = sum(len(loaders[t]["train"]) for t in tasks)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    history, best_score, t_start = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        schedule = [t for t in tasks for _ in range(len(loaders[t]["train"]))]
        random.shuffle(schedule)
        iters = {t: iter(loaders[t]["train"]) for t in tasks}
        running = {t: 0.0 for t in tasks}

        for step, task in enumerate(schedule, 1):
            batch = next(iters[task])
            intent_logits, emotion_logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            targets = {f"{task}_targets": batch["labels"].to(device)}
            losses = loss_fn(intent_logits, emotion_logits, **targets)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running[task] += losses[task].item()
            if step % args.log_every == 0:
                elapsed = time.time() - t_start
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} task={task} "
                      f"loss={losses[task].item():.4f} elapsed={elapsed / 60:.1f}m", flush=True)

        val = {t: evaluate(model, loaders[t]["validation"], t, device) for t in tasks}
        score = float(np.mean([val[t]["macro_f1"] for t in tasks]))
        train_loss = {t: round(running[t] / len(loaders[t]["train"]), 4) for t in tasks}
        history.append({"epoch": epoch, "train_loss": train_loss, "validation": val})
        print(f"Epoch {epoch}/{args.epochs} train_loss={train_loss} val={val}", flush=True)

        if score > best_score:
            best_score = score
            model.save_pretrained(args.output_dir, extra={"tasks": tasks, "best_epoch": epoch})
            tokenizer.save_pretrained(args.output_dir)

    train_minutes = round((time.time() - t_start) / 60, 1)

    # ── Test the best checkpoint ─────────────────────────────────────────
    model = MTLIntentEmotionModel.from_pretrained(args.output_dir).to(device)
    test = {t: evaluate(model, loaders[t]["test"], t, device) for t in tasks}
    best_epoch = max(history, key=lambda h: np.mean([h["validation"][t]["macro_f1"] for t in tasks]))["epoch"]

    metrics = {
        "tasks": tasks,
        "args": vars(args),
        "best_epoch": best_epoch,
        "test": test,
        "history": history,
        "train_minutes": train_minutes,
        "device": device,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nBest epoch {best_epoch} — test: {test}")
    print(f"Saved model + metrics to {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default="intent,emotion", help="comma-separated subset of: intent,emotion")
    p.add_argument("--encoder", default="distilroberta-base")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--freeze_layers", type=int, default=0, help="freeze embeddings + bottom N encoder layers")
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--max_train_per_task", type=int, default=None, help="cap train size per task (seeded sample)")
    p.add_argument("--max_eval_per_task", type=int, default=None, help="cap val size per task (seeded sample)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--output_dir", default="runs/mtl")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
