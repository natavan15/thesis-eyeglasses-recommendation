import csv
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay

from computer_vision_pipeline.src.data.dataset import load_train_val_test_dataloaders
from computer_vision_pipeline.src.models.model import create_model, get_device, CLASS_NAMES
from computer_vision_pipeline.src.utils.paths import DATA_DIR, MODELS_DIR, RESULTS_DIR


def export_misclassified_examples(model, loader, device, output_csv_path, max_errors=100):
    model.eval()
    rows = []

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for i in range(len(labels)):
                pred_idx = preds[i].item()
                true_idx = labels[i].item()

                if pred_idx != true_idx:
                    rows.append({
                        "Image_ID": paths[i].split("\\")[-1],
                        "Image_Path": paths[i],
                        "True_Label": CLASS_NAMES[true_idx],
                        "Predicted_Label": CLASS_NAMES[pred_idx],
                        "Correct?": 0,
                    })

                    if len(rows) >= max_errors:
                        break

            if len(rows) >= max_errors:
                break

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Image_ID", "Image_Path", "True_Label", "Predicted_Label", "Correct?"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} misclassified examples to: {output_csv_path}")


def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds


def save_text_results(all_labels, all_preds, accuracy, weighted_f1, output_txt_path):
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
    cm = confusion_matrix(all_labels, all_preds)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("Cross-Dataset Evaluation Results\n")
        f.write("=" * 35 + "\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Weighted F1-score: {weighted_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n")

    print(f"Saved text results to: {output_txt_path}")


def save_confusion_matrix_plot(all_labels, all_preds, output_png_path):
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix: Dataset1 -> Dataset2")
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved confusion matrix plot to: {output_png_path}")


def main():
    device = get_device()
    print("Using device:", device)

    model_path = MODELS_DIR / "dataset1_inception_v4_facecrop.pth"
    print("Loading model from:", model_path)

    dataset2_train_dir = DATA_DIR / "processed_faces" / "dataset2" / "train"
    dataset2_val_dir = DATA_DIR / "processed_faces" / "dataset2" / "val"
    dataset2_test_dir = DATA_DIR / "processed_faces" / "dataset2" / "test"

    _, _, test_loader = load_train_val_test_dataloaders(
        train_dir=dataset2_train_dir,
        val_dir=dataset2_val_dir,
        test_dir=dataset2_test_dir,
        batch_size=16,
    )

    model = create_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_csv_path = RESULTS_DIR / "misclassified_examples_dataset1_to_dataset2_facecrop.csv"
    output_txt_path = RESULTS_DIR / "evaluation_results_dataset1_to_dataset2_facecrop.txt"
    output_png_path = RESULTS_DIR / "confusion_matrix_dataset1_to_dataset2_facecrop.png"

    export_misclassified_examples(
        model=model,
        loader=test_loader,
        device=device,
        output_csv_path=output_csv_path,
        max_errors=100,
    )

    all_labels, all_preds = evaluate_model(model, test_loader, device)

    accuracy = sum(int(p == y) for p, y in zip(all_preds, all_labels)) / len(all_labels)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"\nCross-dataset Accuracy: {accuracy:.4f}")
    print(f"Cross-dataset Weighted F1-score: {weighted_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    save_text_results(
        all_labels=all_labels,
        all_preds=all_preds,
        accuracy=accuracy,
        weighted_f1=weighted_f1,
        output_txt_path=output_txt_path,
    )

    save_confusion_matrix_plot(
        all_labels=all_labels,
        all_preds=all_preds,
        output_png_path=output_png_path,
    )


if __name__ == "__main__":
    main()