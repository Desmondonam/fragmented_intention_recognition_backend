"""
Quick Demo Script
==================
Run this to verify the full pipeline works locally — no server needed.

Usage:
    python demo.py

This spins up the model with random weights (no training required)
and runs a simulated multi-turn conversation through the escalation engine.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from model.mtl_model import MTLIntentEmotionModel, INTENT_LABELS, EMOTION_LABELS
from app.inference import MTLInferencePipeline

# ── ANSI colours for readable terminal output ──────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

ACTION_COLOURS = {
    "autonomous":      GREEN,
    "flag_for_review": YELLOW,
    "escalate_now":    RED,
}

# ── Simulated conversation ─────────────────────────────────────────────────
CONVERSATION = [
    "Hi, what's my account balance?",
    "Hmm, I was also wondering if I can update my address.",
    "Actually, my card was stolen. I need to cancel it.",
    "I've been on hold for 20 minutes, nobody is helping me.",
    "This is completely ridiculous. I need to speak to a human RIGHT NOW.",
]

def print_separator():
    print("─" * 65)

def run_demo():
    print(f"\n{BOLD}{'='*65}")
    print("  MTL INTENT & EMOTION DEMO  —  Multi-Turn Conversation")
    print(f"{'='*65}{RESET}\n")

    # Load model (random weights for demo; in production load from checkpoint)
    model = MTLIntentEmotionModel(encoder_name="roberta-base", freeze_encoder_layers=0)
    pipeline = MTLInferencePipeline(model=model, encoder_name="roberta-base")

    print(f"{CYAN}Model device: {pipeline.device}{RESET}")
    print(f"{CYAN}Intent labels ({len(INTENT_LABELS)}): {INTENT_LABELS}{RESET}")
    print(f"{CYAN}Emotion labels ({len(EMOTION_LABELS)}): {EMOTION_LABELS}{RESET}\n")
    print_separator()

    for utterance in CONVERSATION:
        result = pipeline.predict(utterance)
        esc = result.escalation
        colour = ACTION_COLOURS.get(esc.action, RESET)

        print(f"\n{BOLD}Turn {result.conversation_turn}{RESET}")
        print(f"  User: \"{utterance}\"")
        print(f"\n  {BOLD}Intents Detected:{RESET}")
        if result.intents:
            for intent in result.intents:
                bar = "█" * int(intent.probability * 20)
                print(f"    [{bar:<20}] {intent.probability:.2f}  {intent.label}")
        else:
            print("    (none above threshold)")

        print(f"\n  {BOLD}Emotion:{RESET}")
        bar = "█" * int(result.emotion.probability * 20)
        print(f"    [{bar:<20}] {result.emotion.probability:.2f}  {result.emotion.label}")

        print(f"\n  {BOLD}Escalation:{RESET}")
        print(f"    Urgency Score : {esc.urgency_score:.4f}")
        print(f"    Action        : {colour}{BOLD}{esc.action.upper()}{RESET}")
        print(f"    Bot Response  : \"{esc.message}\"")
        print_separator()

    print(f"\n{GREEN}Demo complete. The model is using random weights — train it with train.py for real predictions.{RESET}\n")


if __name__ == "__main__":
    run_demo()
