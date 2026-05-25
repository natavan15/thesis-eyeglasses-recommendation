import torch
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from computer_vision_pipeline.src.data.dataset import load_train_val_test_dataloaders
from computer_vision_pipeline.src.models.model import create_model, get_device
from computer_vision_pipeline.src.utils.paths import DATA_DIR, MODELS_DIR, RESULTS_DIR


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, labels, paths in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def plot_training_curves(train_accuracies, val_accuracies, train_losses, output_path):
    epochs_range = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs_range,
        train_accuracies,
        marker="o",
        linewidth=2.5,
        label="Train Accuracy"
    )

    plt.plot(
        epochs_range,
        val_accuracies,
        marker="o",
        linewidth=2.5,
        linestyle="--",
        label="Validation Accuracy"
    )

    plt.title("Training vs Validation Accuracy", fontsize=18)
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved training curve to: {output_path}")


def main():
    device = get_device()
    print("Using device:", device)

    train_dir = DATA_DIR / "processed_faces" / "dataset1" / "train"
    val_dir = DATA_DIR / "processed_faces" / "dataset1" / "val"
    test_dir = DATA_DIR / "processed_faces" / "dataset1" / "test"

    print("Train dir:", train_dir)
    print("Val dir:", val_dir)
    print("Test dir:", test_dir)

    train_loader, val_loader, test_loader = load_train_val_test_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=16,
    )

    model = create_model(pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5

    train_accuracies = []
    val_accuracies = []
    train_losses = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")

    test_acc = evaluate(model, test_loader, device)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "dataset1_inception_v4_facecrop.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    plot_path = RESULTS_DIR / "train_val_accuracy.png"
    plot_training_curves(
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        train_losses=train_losses,
        output_path=plot_path,
    )


if __name__ == "__main__":
    main()