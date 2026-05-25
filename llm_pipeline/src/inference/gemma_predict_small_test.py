import os
import csv
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = r"D:\Thesis\thesis-eyeglasses\llm_pipeline\models\gemma4"
dataset_path = r"D:\Thesis\thesis-eyeglasses\llm_pipeline\data\github_test_dataset_faces"
output_csv = r"D:\Thesis\thesis-eyeglasses\llm_pipeline\results\gemma_small_test_predictions_faces.csv"

classes = ["heart", "oblong", "oval", "round", "square"]

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

prompt = """
You are a strict face-shape classification assistant for a master's thesis experiment.

Your task:
Analyze the visible face geometry in the image and classify the face into exactly ONE of these five classes:

heart, oblong, oval, round, square

Important:
Do not guess based on beauty, gender, hairstyle, expression, skin color, or age.
Focus only on facial geometry.

Analyze these visual features:
1. Face length compared with face width
2. Forehead width
3. Cheekbone width
4. Jawline shape
5. Chin shape
6. Whether the face looks soft/rounded or angular
7. Whether the face is clearly elongated

Class definitions:

heart:
- Forehead is wider than jaw
- Cheekbones may be wide
- Chin looks narrow or pointed
- Lower face becomes narrower

oblong:
- Face is clearly longer than it is wide
- Forehead, cheeks, and jaw may have similar width
- Face looks vertically elongated
- Chin/jaw may look longer

oval:
- Face is slightly longer than wide
- Forehead and jaw are balanced
- Jawline is softly rounded
- Chin is not very pointed
- Face is balanced, but not extremely long

round:
- Face width and face length are similar
- Cheeks look full or rounded
- Jawline is soft, not angular
- Chin is rounded
- Face does not look long

square:
- Forehead, cheeks, and jaw have similar width
- Jawline looks strong, broad, or angular
- Chin area looks wider or flatter
- Face has clear angles

Decision rules:
- Do NOT choose oval as a default answer.
- Choose oval only if the face is clearly balanced and softly rounded.
- If the face is much longer than wide, choose oblong.
- If the jaw is strong/angular, choose square.
- If the face width and length are similar with soft cheeks, choose round.
- If the forehead is wider and chin is narrow/pointed, choose heart.
- If uncertain, still choose the closest class, but mark confidence as low.

Before final answer, mentally compare all five classes.
Return only the final structured answer.

Output format exactly:

Face shape: <heart/oblong/oval/round/square>
Reason: <short explanation based on geometry>
Confidence: <low/medium/high>
"""

rows = []

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
            max_new_tokens=25,
            do_sample=False
        )

    input_length = inputs["input_ids"].shape[-1]
    response = processor.decode(output[0][input_length:], skip_special_tokens=True)

    print(response)

    rows.append({
        "image_path": image_path,
        "image_name": image_name,
        "true_label": true_class,
        "gemma_response": response
    })

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["image_path", "image_name", "true_label", "gemma_response"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved results to: {output_csv}")