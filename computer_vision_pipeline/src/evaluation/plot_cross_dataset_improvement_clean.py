import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from computer_vision_pipeline.src.utils.paths import RESULTS_DIR


def main():
    # Data
    stages = ["Baseline", "Face Detection", "Face Detection + Training"]
    x = np.arange(len(stages))

    accuracy = [0.6000, 0.6267, 0.7467]
    f1_score = [0.6050, 0.6238, 0.7522]

    # Figure
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Lines
    ax.plot(x, accuracy, marker="o", linewidth=2.5, label="Accuracy")
    ax.plot(x, f1_score, marker="o", linewidth=2.5, linestyle="--", label="Weighted F1-score")

    # Labels
    ax.set_title("Cross-Dataset Performance Improvement", fontsize=18, weight="bold")
    ax.set_xlabel("Experiment Stage", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylim(0.58, 0.78)

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    # Value labels
    for i, v in enumerate(accuracy):
        ax.text(i, v - 0.008, f"{v:.4f}", ha="center", fontsize=10)

    for i, v in enumerate(f1_score):
        ax.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=10)

    # Key annotation (clean and minimal)
    improvement = accuracy[-1] - accuracy[0]

    ax.annotate(
        f"+{improvement:.4f} improvement",
        xy=(2, accuracy[2]),
        xytext=(1.6, 0.69),
        arrowprops=dict(arrowstyle="->", lw=1.5),
        fontsize=11
    )

    plt.tight_layout()

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RESULTS_DIR / "cross_dataset_improvement_clean.png"
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()