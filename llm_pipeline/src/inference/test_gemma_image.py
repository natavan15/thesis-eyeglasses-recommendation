from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

# -----------------------------
# Paths
# -----------------------------
model_path = r"D:\Thesis\thesis-eyeglasses\llm_pipeline\models\gemma4"

# Change this filename if your image has another name or extension
image_path = r"D:\Thesis\thesis-eyeglasses\llm_pipeline\data\test_images\test_face_1.png"

# -----------------------------
# Load model and processor
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
# Load image
# -----------------------------
print("Loading image...")
image = Image.open(image_path).convert("RGB")

# -----------------------------
# Prompt
# -----------------------------
prompt = """
You are an eyewear recommendation assistant for a master's thesis experiment.

Your task is to analyze the person's face in the image and provide a structured eyewear recommendation.

Please do the following:

1. Estimate the most likely face shape from only these classes:
   - round
   - oval
   - square
   - heart
   - oblong

2. Briefly explain why you selected that face shape.
   Consider visible facial features such as:
   - face length and width
   - jawline
   - forehead width
   - cheekbone area
   - chin shape

3. Recommend suitable eyeglass frame styles for this face shape.

4. Mention which frame styles should be avoided, if any.

5. Be careful and avoid overclaiming.
   If the image is unclear, say that the prediction is uncertain.

Return the answer in this exact format:

Face shape:
Reason:
Recommended frames:
Frames to avoid:
Confidence:
"""

# -----------------------------
# Prepare message
# -----------------------------
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

# -----------------------------
# Generate response
# -----------------------------
print("Generating response...")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=False
    )

response = processor.decode(output[0], skip_special_tokens=True)

print("\nGemma Response:\n")
print(response)