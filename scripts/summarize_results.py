"""
Build the multi-task vs single-task comparison table from results/*.json.

    python -m scripts.summarize_results   # writes results/RESULTS.md
"""

import json
import os

RESULTS_DIR = "results"


def load(name):
    with open(os.path.join(RESULTS_DIR, f"{name}.json")) as f:
        return json.load(f)


def main():
    mtl, stl = load("mtl"), {"intent": load("stl_intent"), "emotion": load("stl_emotion")}
    names = {"intent": "Intent (Banking77, 77 classes)", "emotion": "Emotion (GoEmotions → Ekman, 7 classes)"}

    lines = [
        "| Task | Metric | Single-task | Multi-task | Δ |",
        "|------|--------|-------------|------------|---|",
    ]
    for task in ("intent", "emotion"):
        for metric, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")):
            s, m = stl[task]["test"][task][metric], mtl["test"][task][metric]
            lines.append(f"| {names[task]} | {label} | {s:.4f} | {m:.4f} | {m - s:+.4f} |")

    a = mtl["args"]
    test_sizes = f"intent n={mtl['test']['intent']['n']}, emotion n={mtl['test']['emotion']['n']}"
    footer = [
        "",
        f"Test sets: {test_sizes}. Encoder `{a['encoder']}`, embeddings + bottom {a['freeze_layers']} layers "
        f"frozen, {a['max_train_per_task']} train examples/task, {a['epochs']} epochs, lr {a['lr']}, "
        f"batch {a['batch_size']}, seed {a['seed']}. Best epoch chosen on validation macro-F1.",
    ]
    table = "\n".join(lines + footer) + "\n"
    with open(os.path.join(RESULTS_DIR, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write(table)
    print(table.encode("ascii", "replace").decode())  # Windows consoles choke on Δ / →


if __name__ == "__main__":
    main()
