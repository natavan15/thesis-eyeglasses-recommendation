import torch
import csv
from datasets import load_from_disk
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from PIL import Image
from tqdm import tqdm

model_path = "/workspace/models/gemma4"
lora_path = "/workspace/lora_face_shape_gemma4_final_shuffled"   # RAW model
val_path = "/workspace/kaggle_validation_dataset"

output_csv = "/workspace/results_llm_raw_validation.csv"

print("Loading dataset...")
dataset = load_from_disk(val_path)

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

correct = 0

print("Running RAW evaluation and saving CSV...")

with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
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

        if pred not in ["heart", "oblong", "oval", "round", "square"]:
            pred = "unknown"

        writer.writerow([example["image_path"], true_label, pred])

        if pred == true_label:
            correct += 1

accuracy = correct / len(dataset)

print("RAW Validation Accuracy:", accuracy)
print("CSV saved to:", output_csv)
