"""
Quick Demo Script
==================
Run this to verify the full pipeline works locally — no server needed.

Usage:
    python demo.py                    # downloads the trained model from the HF Hub
    MODEL_DIR=runs/mtl python demo.py # or use a local training run

Runs a simulated multi-turn conversation through the trained model and the
escalation engine.
"""

from app.api import load_pipeline_from_env

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
    "Hi, when will my new card arrive?",
    "Hmm, I was also wondering if I can update my address.",
    "Actually, my card was stolen. I need to block it.",
    "I've been on hold for 20 minutes, nobody is helping me.",
    "This is completely ridiculous. I need to speak to a human RIGHT NOW.",
]

def print_separator():
    print("─" * 65)

def run_demo():
    print(f"\n{BOLD}{'='*65}")
    print("  MTL INTENT & EMOTION DEMO  —  Multi-Turn Conversation")
    print(f"{'='*65}{RESET}\n")

    pipeline = load_pipeline_from_env()

    print(f"{CYAN}Model: {pipeline.model.encoder_name} on {pipeline.device}{RESET}")
    print(f"{CYAN}{len(pipeline.intent_labels)} intents, emotions: {pipeline.emotion_labels}{RESET}\n")
    print_separator()

    for utterance in CONVERSATION:
        result = pipeline.predict(utterance)
        esc = result.escalation
        colour = ACTION_COLOURS.get(esc.action, RESET)

        print(f"\n{BOLD}Turn {result.conversation_turn}{RESET}")
        print(f"  User: \"{utterance}\"")
        print(f"\n  {BOLD}Top Intents:{RESET}")
        for intent in result.intents:
            bar = "█" * int(intent.probability * 20)
            print(f"    [{bar:<20}] {intent.probability:.2f}  {intent.label}")

        print(f"\n  {BOLD}Emotion:{RESET}")
        bar = "█" * int(result.emotion.probability * 20)
        print(f"    [{bar:<20}] {result.emotion.probability:.2f}  {result.emotion.label}")

        print(f"\n  {BOLD}Escalation:{RESET}")
        print(f"    Urgency Score : {esc.urgency_score:.4f}")
        print(f"    Action        : {colour}{BOLD}{esc.action.upper()}{RESET}")
        print(f"    Bot Response  : \"{esc.message}\"")
        print_separator()

    print(f"\n{GREEN}Demo complete.{RESET}\n")


if __name__ == "__main__":
    run_demo()
