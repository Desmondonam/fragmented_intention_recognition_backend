| Task | Metric | Single-task | Multi-task | Δ |
|------|--------|-------------|------------|---|
| Intent (Banking77, 77 classes) | Accuracy | 0.8760 | 0.8714 | -0.0046 |
| Intent (Banking77, 77 classes) | Macro-F1 | 0.8740 | 0.8687 | -0.0053 |
| Emotion (GoEmotions → Ekman, 7 classes) | Accuracy | 0.6826 | 0.6739 | -0.0087 |
| Emotion (GoEmotions → Ekman, 7 classes) | Macro-F1 | 0.5928 | 0.5641 | -0.0287 |

Test sets: intent n=3080, emotion n=4968. Encoder `distilroberta-base`, embeddings + bottom 3 layers frozen, 9000 train examples/task, 3 epochs, lr 5e-05, batch 32, seed 42. Best epoch chosen on validation macro-F1.
