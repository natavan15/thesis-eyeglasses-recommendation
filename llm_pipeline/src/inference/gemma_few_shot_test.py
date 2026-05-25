import os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# Paths
model_path = r"D:\Thesis\thesis-eyeglasses\llm_pipeline\models\gemma4"
dataset_path = r"D:\Thesis\thesis-eyeglasses\llm_pipeline\data\github_test_dataset_faces"
few_shot_path = r"D:\Thesis\thesis-eyeglasses\llm_pipeline\data\few_shot_examples"

classes = ["heart", "oblong", "oval", "round", "square"]

# Load model
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

# Load few-shot images
examples = []
for cls in classes:
    folder = os.path.join(few_shot_path, cls)
    img_name = os.listdir(folder)[0]
    img_path = os.path.join(folder, img_name)
    img = Image.open(img_path).convert("RGB")

    examples.append({
        "type": "image",
        "image": img
    })
    examples.append({
        "type": "text",
        "text": f"This is an example of a {cls} face shape."
    })

# Prompt
instruction = """
Now analyze the next face image.

Classify it into one of:
heart, oblong, oval, round, square.

Return ONLY:

Face shape: <one label>
"""

# Test 1 image per class
for true_class in classes:
    class_folder = os.path.join(dataset_path, true_class)
    image_name = os.listdir(class_folder)[0]
    image_path = os.path.join(class_folder, image_name)

    print(f"\nProcessing: {true_class}/{image_name}")

    test_image = Image.open(image_path).convert("RGB")

    content = []
    content.extend(examples)

    content.append({
        "type": "image",
        "image": test_image
    })
    content.append({
        "type": "text",
        "text": instruction
    })

    messages = [
        {
            "role": "user",
            "content": content
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
            max_new_tokens=50,
            do_sample=False
        )

    input_length = inputs["input_ids"].shape[-1]
    response = processor.decode(
        output[0][input_length:],
        skip_special_tokens=True
    )

    print("Prediction:", response)