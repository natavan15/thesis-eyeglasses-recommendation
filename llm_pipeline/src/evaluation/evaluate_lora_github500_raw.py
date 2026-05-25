import csv
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


# =========================
# Paths
# =========================

model_path = r"D:\Thesis\thesis-eyeglasses\llm_pipeline\models\gemma4"

lora_path = r"D:\Thesis\thesis-eyeglasses\llm_pipeline\models\lora_raw\lora_face_shape_gemma4_final_shuffled"

data_path = Path(r"D:\Thesis\thesis-eyeglasses\openai_vision_pipeline\data\github_raw_500")

output_csv = r"D:\Thesis\thesis-eyeglasses\llm_pipeline\results\lora\results_gemma_lora_raw_test500.csv"


# =========================
# Settings
# =========================

classes = ["heart", "oblong", "oval", "round", "square"]
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

Path(output_csv).parent.mkdir(parents=True, exist_ok=True)


# =========================
# Load image paths from folders
# =========================

print("Loading GitHub RAW dataset from class folders...")

dataset = []

for cls in classes:
    class_folder = data_path / cls

    if not class_folder.exists():
        print(f"Warning: class folder not found: {class_folder}")
        continue

    for img_path in class_folder.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in image_extensions:
            dataset.append({
                "image_path": str(img_path),
                "true_label": cls
            })

print(f"Loaded {len(dataset)} images.")


# =========================
# Load model and LoRA adapter
# =========================

print("Loading processor...")
processor = AutoProcessor.from_pretrained(lora_path)

print("Loading base Gemma 4 model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

dtype = torch.float16 if device == "cuda" else torch.float32

base_model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=dtype,
    low_cpu_mem_usage=False
)

model = PeftModel.from_pretrained(base_model, lora_path)
model = model.to(device)
model.eval()


# =========================
# Prompt
# =========================

prompt_text = """
Classify the face shape into exactly one of:
heart, oblong, oval, round, square.

Return only one word.
"""


# =========================
# Prediction helper
# =========================

def clean_prediction(text):
    text = text.strip().lower()

    for cls in classes:
        if cls in text:
            return cls

    return "unknown"


# =========================
# Run inference
# =========================

y_true = []
y_pred = []

print("Running LoRA Gemma 4 RAW test on GitHub 500 images...")

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "true_label", "predicted_label"])

    for example in tqdm(dataset):
        image_path = example["image_path"]
        true_label = example["true_label"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Could not open image: {image_path} | Error: {e}")
            continue

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}
            ],
        }]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False
            )

        input_len = inputs["input_ids"].shape[-1]

        raw_pred = processor.decode(
            output[0][input_len:],
            skip_special_tokens=True
        ).strip().lower()

        pred = clean_prediction(raw_pred)

        writer.writerow([image_path, true_label, pred])

        y_true.append(true_label)
        y_pred.append(pred)


# =========================
# Evaluation
# =========================

accuracy = accuracy_score(y_true, y_pred)
weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    labels=classes,
    zero_division=0
)

print("\nLoRA Gemma 4 RAW TEST RESULTS")
print("Total evaluated images:", len(y_true))
print("Accuracy:", round(accuracy, 4))
print("Weighted F1:", round(weighted_f1, 4))
print("CSV saved to:", output_csv)