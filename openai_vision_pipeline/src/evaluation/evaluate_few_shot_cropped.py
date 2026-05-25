import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

INPUT_FILE = "openai_vision_pipeline/results/few_shot_cropped75.xlsx"
OUTPUT_TXT = "openai_vision_pipeline/results/few_shot_cropped75_evaluation.txt"
OUTPUT_CM = "openai_vision_pipeline/results/few_shot_cropped75_confusion_matrix.png"

valid_labels = ["heart", "oblong", "oval", "round", "square"]

df = pd.read_excel(INPUT_FILE)

df["true_label"] = df["true_label"].astype(str).str.lower().str.strip()
df["predicted_label"] = df["predicted_label"].astype(str).str.lower().str.strip()

df = df[df["true_label"].isin(valid_labels)]
df = df[df["predicted_label"].isin(valid_labels)]

y_true = df["true_label"]
y_pred = df["predicted_label"]

acc = accuracy_score(y_true, y_pred)

report = classification_report(
    y_true,
    y_pred,
    labels=valid_labels,
    zero_division=0
)

cm = confusion_matrix(y_true, y_pred, labels=valid_labels)

print("Accuracy:", acc)
print(report)

os.makedirs("openai_vision_pipeline/results", exist_ok=True)

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=valid_labels,
    yticklabels=valid_labels,
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Few-shot ChatGPT CROPPED 75 Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_CM)
plt.close()

print("Saved evaluation and confusion matrix.")