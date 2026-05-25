import os
import csv
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# -----------------------------
# Paths
# -----------------------------
model_path = r"D:\Thesis\thesis-eyeglasses\llm_pipeline\models\gemma4"
dataset_path = r"D:\Thesis\thesis-eyeglasses\llm_pipeline\data\github_test_dataset_faces"
output_csv = r"D:\Thesis\thesis-eyeglasses\llm_pipeline\results\gemma_small_test_scoring_faces.csv"

classes = ["heart", "oblong", "oval", "round", "square"]

# -----------------------------
# Load model
# -----------------------------
print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_path)

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)

model.eval()

# -----------------------------
# Scoring prompt
# -----------------------------
prompt = """
You are a strict face-shape evaluator for an eyewear recommendation thesis.

Look only at the visible facial geometry:
- face length vs width
- forehead width
- cheekbone width
- jawline shape
- chin shape
- roundness or angularity

Score EACH face shape from 0 to 10.

Meaning:
0 = clearly not this face shape
5 = possible but uncertain
10 = very strong match

Face shape definitions:
heart:
- forehead is noticeably wider than jaw
- cheekbones may be prominent
- jawline narrows toward chin
- chin is pointed or narrow
- upper face wider than lower face

oblong:
- face is significantly longer than wide
- forehead, cheekbones, and jaw have similar width
- cheeks are not very full
- overall elongated vertical appearance

oval:
- face slightly longer than wide (not extreme)
- forehead slightly wider than jaw
- jawline is soft and rounded
- chin is gently curved, not sharp
- overall balanced proportions

round:
- face width ≈ face length
- cheeks are full and prominent
- jawline is soft and not angular
- chin is rounded
- overall circular appearance

square:
- forehead, cheekbones, and jaw have similar width
- jawline is strong, wide, and angular
- chin appears flatter or broader (not pointed)
- overall boxy/angular appearance


Important:
Do not give all classes similar scores.
Do not automatically give oval the highest score.
Compare all five classes carefully.

Return exactly this format:

heart: <0-10>
oblong: <0-10>
oval: <0-10>
round: <0-10>
square: <0-10>
best_class: <heart/oblong/oval/round/square>
reason: <one short sentence>
"""

rows = []

# -----------------------------
# Test one image per class
# -----------------------------
for true_class in classes:
    class_folder = os.path.join(dataset_path, true_class)

    image_files = sorted([
        f for f in os.listdir(class_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    image_name = image_files[0]
    image_path = os.path.join(class_folder, image_name)

    print(f"\nProcessing: {true_class}/{image_name}")

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False
        )

    input_length = inputs["input_ids"].shape[-1]
    response = processor.decode(
        output[0][input_length:],
        skip_special_tokens=True
    )

    print(response)

    rows.append({
        "image_path": image_path,
        "image_name": image_name,
        "true_label": true_class,
        "gemma_response": response
    })

# -----------------------------
# Save results
# -----------------------------
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["image_path", "image_name", "true_label", "gemma_response"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved results to: {output_csv}")