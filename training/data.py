"""
Dataset loading for the two tasks.

  intent  — Banking77 (PolyAI; parquet copy at legacy-datasets/banking77): 13,083 customer-service queries,
            77 intents. It ships only train/test, so 10% of train is held out
            (stratified, fixed seed) as a validation split for model selection.
  emotion — GoEmotions (google-research-datasets/go_emotions, "simplified"):
            Reddit comments with 27 emotions + neutral. Labels are folded into
            Ekman's 7 classes and only comments whose labels all map to one
            Ekman class are kept, giving a clean single-label task.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from model.labels import EMOTION_LABELS, GOEMOTIONS_TO_EKMAN, INTENT_LABELS

TASKS = ("intent", "emotion")


@dataclass
class Split:
    texts: List[str]
    labels: List[int]

    def __len__(self):
        return len(self.texts)

    def sample(self, n: Optional[int], seed: int = 42) -> "Split":
        """Seeded random subset of size n (or the whole split if n is None or too large)."""
        if n is None or n >= len(self):
            return self
        idx = sorted(random.Random(seed).sample(range(len(self)), n))
        return Split([self.texts[i] for i in idx], [self.labels[i] for i in idx])


def load_banking77(val_fraction: float = 0.1, seed: int = 42) -> Dict[str, Split]:
    # PolyAI/banking77 is script-based, which current `datasets` no longer runs;
    # legacy-datasets/banking77 is the Hub-maintained parquet copy of the same data.
    ds = load_dataset("legacy-datasets/banking77")
    names = ds["train"].features["label"].names
    assert names == INTENT_LABELS, "Banking77 label order differs from model.labels.INTENT_LABELS"

    tr_x, va_x, tr_y, va_y = train_test_split(
        ds["train"]["text"], ds["train"]["label"],
        test_size=val_fraction, random_state=seed, stratify=ds["train"]["label"],
    )
    return {
        "train": Split(list(tr_x), list(tr_y)),
        "validation": Split(list(va_x), list(va_y)),
        "test": Split(list(ds["test"]["text"]), list(ds["test"]["label"])),
    }


def goemotions_to_ekman_label(label_ids: List[int], names: List[str]) -> Optional[int]:
    """Return the Ekman class index if all labels agree on one class, else None."""
    ekman = {GOEMOTIONS_TO_EKMAN[names[i]] for i in label_ids}
    return EMOTION_LABELS.index(ekman.pop()) if len(ekman) == 1 else None


def load_goemotions_ekman() -> Dict[str, Split]:
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    names = ds["train"].features["labels"].feature.names

    splits = {}
    for split in ("train", "validation", "test"):
        texts, labels = [], []
        for text, label_ids in zip(ds[split]["text"], ds[split]["labels"], strict=True):
            y = goemotions_to_ekman_label(label_ids, names)
            if y is not None:
                texts.append(text)
                labels.append(y)
        splits[split] = Split(texts, labels)
    return splits


def load_task(task: str, seed: int = 42) -> Dict[str, Split]:
    if task == "intent":
        return load_banking77(seed=seed)
    if task == "emotion":
        return load_goemotions_ekman()
    raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}")


class TextClassificationDataset(Dataset):
    def __init__(self, split: Split):
        self.split = split

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        return self.split.texts[idx], self.split.labels[idx]


def make_collate_fn(tokenizer, max_length: int = 64):
    """Tokenize with dynamic padding — much faster than padding every batch to max_length."""

    def collate(batch):
        texts, labels = zip(*batch, strict=True)
        enc = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate
