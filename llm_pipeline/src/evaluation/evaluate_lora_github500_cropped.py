import csv
import torch
from datasets import load_from_disk
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

model_path = "/workspace/models/gemma4"
lora_path = "/workspace/lora_face_shape_gemma4_final_cropped"   # CROPPED MODEL
data_path = "/workspace/github_test_dataset"
output_csv = "/workspace/results_llm_cropped_test500.csv"

classes = ["heart", "oblong", "oval", "round", "square"]

print("Loading GitHub dataset (500)...")
dataset = load_from_disk(data_path)

processor = AutoProcessor.from_pretrained(lora_path)

base_model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

prompt_text = """
Classify the face shape into exactly one of:
heart, oblong, oval, round, square.

Return only one word.
"""

y_true = []
y_pred = []

print("Running CROPPED LLM test on 500 images...")

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "true_label", "predicted_label"])

    for example in tqdm(dataset):
        image = Image.open(example["image_path"]).convert("RGB")
        true_label = example["label"]

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
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False
            )

        input_len = inputs["input_ids"].shape[-1]
        pred = processor.decode(output[0][input_len:], skip_special_tokens=True).strip().lower()

        if pred not in classes:
            pred = "unknown"

        writer.writerow([example["image_path"], true_label, pred])

        y_true.append(true_label)
        y_pred.append(pred)

accuracy = accuracy_score(y_true, y_pred)
weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=classes)

print("\nCROPPED LLM TEST (500) RESULTS")
print("Accuracy:", round(accuracy, 4))
print("Weighted F1:", round(weighted_f1, 4))
print("CSV saved to:", output_csv)
