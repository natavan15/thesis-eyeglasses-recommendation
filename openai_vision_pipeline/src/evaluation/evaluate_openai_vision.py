import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("openai_vision_pipeline/results/openai_vision_cropped75.csv")

# Clean missing predictions
df["predicted_label"] = df["predicted_label"].fillna("")

# Filter only valid predictions
valid_labels = ["heart", "oblong", "oval", "round", "square"]
df = df[df["predicted_label"].isin(valid_labels)]

y_true = df["true_label"]
y_pred = df["predicted_label"]

# Accuracy
acc = accuracy_score(y_true, y_pred)

# Classification report
report = classification_report(y_true, y_pred)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=valid_labels)

print("Accuracy:", acc)
print("\nClassification Report:\n", report)

# Save results
with open("openai_vision_pipeline/results/openai_vision_evaluation.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n\n")
    f.write(report)

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=valid_labels,
            yticklabels=valid_labels,
            cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("OpenAI Vision Confusion Matrix")

plt.savefig("openai_vision_pipeline/results/openai_vision_confusion_matrix.png")
plt.close()

print("Saved evaluation results and confusion matrix.")