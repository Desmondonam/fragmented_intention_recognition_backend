"""
Publish a trained run directory to the Hugging Face Hub.

    huggingface-cli login
    python -m scripts.push_to_hub --run_dir runs/mtl --repo_id <user>/intent-emotion-mtl-distilroberta

Uploads the weights, mtl_config.json, tokenizer files and metrics.json, plus a
generated model card. Pin the API to the resulting commit with MODEL_REVISION.
"""

import argparse
import json
import os

from huggingface_hub import HfApi

CARD_TEMPLATE = """---
license: mit
language: en
library_name: pytorch
datasets:
  - legacy-datasets/banking77
  - google-research-datasets/go_emotions
pipeline_tag: text-classification
---

# {repo_name}

Multi-task {encoder} with two heads, trained jointly:

- **Intent**: 77 Banking77 intents
- **Emotion**: 7 Ekman classes (GoEmotions labels folded via the official Ekman mapping)

Used by [intent-emotion-escalation-api](https://github.com/Desmondonam/intent-emotion-escalation-api).

## Test results

| Task | Accuracy | Macro-F1 |
|------|----------|----------|
{rows}

Load with `MTLIntentEmotionModel.from_pretrained(snapshot_download("{repo_id}"))`.
"""


def main(args):
    with open(os.path.join(args.run_dir, "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(args.run_dir, "mtl_config.json")) as f:
        encoder = json.load(f)["encoder_name"]

    rows = "\n".join(
        f"| {task} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} |" for task, m in metrics["test"].items()
    )
    card = CARD_TEMPLATE.format(repo_name=args.repo_id.split("/")[-1], repo_id=args.repo_id,
                                encoder=encoder, rows=rows)
    with open(os.path.join(args.run_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)

    api = HfApi()
    api.create_repo(args.repo_id, exist_ok=True, private=args.private)
    commit = api.upload_folder(folder_path=args.run_dir, repo_id=args.repo_id,
                               commit_message="Upload trained MTL checkpoint")
    print(f"Uploaded to https://huggingface.co/{args.repo_id} ({commit.oid})")
    print(f"Pin the API with MODEL_REPO_ID={args.repo_id} MODEL_REVISION={commit.oid}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default="runs/mtl")
    p.add_argument("--repo_id", required=True)
    p.add_argument("--private", action="store_true")
    main(p.parse_args())
