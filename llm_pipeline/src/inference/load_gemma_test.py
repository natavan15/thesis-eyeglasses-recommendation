from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_path = r"D:\Thesis\thesis-eyeglasses\llm_pipeline\models\gemma4"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_path)

print("Loading model with lower memory...")
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True
)

print("Model loaded successfully!")
print(type(model))