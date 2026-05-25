import matplotlib.pyplot as plt


def main():
    # Replace with your exact values if needed
    train_accuracies = [0.70, 0.85, 0.90, 0.94, 0.96]
    val_accuracies = [0.64, 0.74, 0.76, 0.81, 0.80]
    epochs = [1, 2, 3, 4, 5]

    plt.figure(figsize=(10, 6))

    # Main curves
    plt.plot(
        epochs,
        train_accuracies,
        marker="o",
        linewidth=2.8,
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        val_accuracies,
        marker="o",
        linewidth=2.8,
        linestyle="--",
        label="Validation Accuracy"
    )

    # Highlight best validation epoch
    best_epoch = val_accuracies.index(max(val_accuracies)) + 1
    best_val = max(val_accuracies)
    plt.scatter(best_epoch, best_val, s=90, zorder=5)
    plt.text(best_epoch - 0.45, best_val + 0.035, "Best validation", fontsize=11)

    # Shade overfitting / generalization gap in later epochs
    gap_start_idx = 3   # starts from epoch 4
    plt.fill_between(
        epochs[gap_start_idx:],
        train_accuracies[gap_start_idx:],
        val_accuracies[gap_start_idx:],
        alpha=0.20,
        label="Generalization gap"
    )

    # Small label inside the shaded area
    mid_x = 4.2
    mid_y = (train_accuracies[3] + val_accuracies[3]) / 2
    plt.text(mid_x, mid_y - 0.02, "Generalization gap", fontsize=11)

    # Layout
    plt.title("Training vs Validation Accuracy", fontsize=18)
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    plt.xticks(epochs)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    output_path = "results/train_val_accuracy_professional.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()