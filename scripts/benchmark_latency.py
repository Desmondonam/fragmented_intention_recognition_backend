"""
Latency benchmark
=================
Measures single-request latency (batch size 1, the API's access pattern) for:

  model     — MTLInferencePipeline.predict: tokenize + forward + escalation
  http      — POST /analyze through the full FastAPI stack (in-process client,
              so network time is excluded but validation/serialization is not)
  serving   — (with --single_task_dirs) encoder forward cost of the one
              multi-task model vs running the two single-task models that
              would otherwise be needed to get both predictions

Utterances are drawn from the Banking77 and GoEmotions test sets so input
lengths are realistic.

    MODEL_DIR=runs/mtl python -m scripts.benchmark_latency --n 500 \
        --single_task_dirs runs/stl_intent runs/stl_emotion
"""

import argparse
import json
import os
import platform
import random
import statistics
import time

import torch
from fastapi.testclient import TestClient

from app.api import create_app, load_pipeline_from_env
from app.model_store import load_model_and_tokenizer
from training.data import load_banking77, load_goemotions_ekman


def percentile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


def summarize(samples_ms):
    return {
        "p50_ms": round(percentile(samples_ms, 50), 1),
        "p95_ms": round(percentile(samples_ms, 95), 1),
        "p99_ms": round(percentile(samples_ms, 99), 1),
        "mean_ms": round(statistics.mean(samples_ms), 1),
        "max_ms": round(max(samples_ms), 1),
        "n": len(samples_ms),
    }


@torch.no_grad()
def time_forward(models, tokenizer, text):
    """Latency of tokenizing once and running each model's forward pass."""
    t0 = time.perf_counter()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    for m in models:
        m(enc["input_ids"], enc["attention_mask"])
    return (time.perf_counter() - t0) * 1000


def main(args):
    torch.set_num_threads(args.threads)
    random.seed(0)
    texts = load_banking77()["test"].texts + load_goemotions_ekman()["test"].texts
    texts = random.sample(texts, args.n)

    pipeline = load_pipeline_from_env()

    for t in texts[: args.warmup]:
        pipeline.predict(t)

    model_ms = []
    for t in texts:
        t0 = time.perf_counter()
        pipeline.predict(t, session_id="bench")
        model_ms.append((time.perf_counter() - t0) * 1000)

    http_ms = []
    with TestClient(create_app(pipeline_factory=lambda: pipeline)) as client:
        for t in texts[: args.warmup]:
            client.post("/analyze", json={"text": t})
        for t in texts:
            t0 = time.perf_counter()
            res = client.post("/analyze", json={"text": t, "session_id": "bench"})
            http_ms.append((time.perf_counter() - t0) * 1000)
            assert res.status_code == 200

    result = {
        "model": summarize(model_ms),
        "http": summarize(http_ms),
        "encoder": pipeline.model.encoder_name,
        "device": pipeline.device,
        "torch_threads": args.threads,
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "torch": torch.__version__,
    }

    if args.single_task_dirs:
        stl_models = [load_model_and_tokenizer(d)[0].eval() for d in args.single_task_dirs]
        mtl_model = pipeline.model
        for t in texts[: args.warmup]:
            time_forward([mtl_model, *stl_models], pipeline.tokenizer, t)
        # Interleave per utterance so CPU frequency/thermal drift affects both equally.
        one_ms, two_ms = [], []
        for t in texts:
            one_ms.append(time_forward([mtl_model], pipeline.tokenizer, t))
            two_ms.append(time_forward(stl_models, pipeline.tokenizer, t))
        result["serving"] = {
            "one_multitask_model": summarize(one_ms),
            "two_single_task_models": summarize(two_ms),
        }
    print(json.dumps(result, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--single_task_dirs", nargs=2, metavar=("INTENT_DIR", "EMOTION_DIR"))
    p.add_argument("--output", default="results/latency.json")
    main(p.parse_args())
