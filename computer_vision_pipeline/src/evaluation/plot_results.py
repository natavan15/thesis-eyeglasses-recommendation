import matplotlib.pyplot as plt


def main():
    stages = ["Baseline", "Face Detection", "Face Detection + Training"]
    accuracy = [0.6000, 0.6267, 0.7467]
    weighted_f1 = [0.6050, 0.6238, 0.7522]

    plt.figure(figsize=(10, 6))

    plt.plot(
        stages,
        accuracy,
        marker="o",
        linewidth=3,
        linestyle="-",
        label="Accuracy"
    )

    plt.plot(
        stages,
        weighted_f1,
        marker="o",
        linewidth=3,
        linestyle="--",
        label="Weighted F1-score"
    )

    plt.title("Cross-Dataset Performance Improvement", fontsize=20)
    plt.xlabel("Experiment Stage", fontsize=14)
    plt.ylabel("Score", fontsize=14)

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    plt.ylim(0.58, 0.78)

    output_path = "results/thesis_results_clean.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()