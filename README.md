# Comparative analysis of models for recognizing prices on gas station signs

## Results

| Model                       | Accuracy (%) | ACER (%) | Robustness (%) | Time inf (s) | FPS   | Memory max (MB) | Size of model (MB) | Total |
|-----------------------------|--------------|----------|----------------|--------------|-------|-----------------|--------------------|-------|
| TrOCR small-printed         | 0.00         | 100.00   | 0.00           | 0.318        | 3.24  | 744.18          | 706.04             | 448   |
| TrOCR small-handwritten     | 0.00         | 100.00   | 0.00           | 0.181        | 6.15  | 744.71          | 706.04             | 448   |
| PARSeq (trained 0 epochs)   | 0.00         | 98.48    | 0.00           | 0.016        | 61.27 | 602.05          | 450.00             | 448   |
| PARSeq (trained 10 epochs)  | 1.34         | 70.28    | 0.00           | 0.034        | 29.38 | 862.02          | 450.00             | 448   |
| PARSeq (trained 50 epochs)  | 2.46         | 57.35    | 0.00           | 0.043        | 23.34 | 855.50          | 450.00             | 448   |
| PARSeq (trained 100 epochs) | 25.67        | 43.52    | 79.69          | 0.041        | 24.60 | 844.26          | 450.00             | 448   |
| Florence-2-base             | 29.02        | 100.00   | 39.78          | 2.435        | 0.41  | 1845.30         | 8323.12            | 448   |
| Qwen2.5-VL-3B               | 24.33        | 100.00   | 51.39          | 67.981       | 0.01  | 12578.47        | 7878.69            | 448   |
| EasyOCR                     | 0.22         | 100.00   | 0.00           | 0.419        | 2.40  | 2014.58         | -                  | 448   |

### Metrics Legend

- **Accuracy** — exact match rate (↑ better)
- **ACER** — character error rate (↓ better)
- **Robustness** — accuracy retention on noisy vs clean images (↑ better)
- **Time inf** — inference time per image in seconds (↓ better)
- **FPS** — frames per second (↑ better)
- **Memory max** — peak RAM usage (↓ better)
- **Size of model** — disk space occupied (↓ better)
- **Total** — total number of price checks