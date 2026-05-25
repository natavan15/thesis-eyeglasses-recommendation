import os
import torch
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model

model_path = "/workspace/models/gemma4"
train_path = "/workspace/face_shape_dataset_cropped"
output_dir = "/workspace/lora_face_shape_gemma4_final_cropped"

os.makedirs(output_dir, exist_ok=True)

print("Loading clean training dataset...")
dataset = load_from_disk(train_path)

# CRITICAL: shuffle dataset to avoid last-class bias
dataset = dataset.shuffle(seed=42)

print("Training samples:", len(dataset))
print("First 20 labels after shuffle:", dataset["label"][:20])

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_path)

print("Loading Gemma model...")
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)

model.config.use_cache = False

print("Adding LoRA adapter...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj.linear", "v_proj.linear"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.train()
model.print_trainable_parameters()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

prompt_text = """
Classify the face shape into exactly one of:
heart, oblong, oval, round, square.

Return only one word.
"""

num_epochs = 2
global_step = 0
total_loss = 0.0

print("Starting FINAL shuffled LoRA fine-tuning...")

for epoch in range(num_epochs):
    for example in tqdm(dataset, desc=f"Epoch {epoch+1}/{num_epochs}"):

        image = Image.open(example["image_path"]).convert("RGB")
        label_text = example["label"]

        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ],
            }
        ]

        full_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": label_text}
                ],
            }
        ]

        user_inputs = processor.apply_chat_template(
            user_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        full_inputs = processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        labels = full_inputs["input_ids"].clone()
        user_len = user_inputs["input_ids"].shape[-1]

        # Train only on assistant answer, not on prompt/image text
        labels[:, :user_len] = -100

        outputs = model(**full_inputs, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        global_step += 1

        if global_step % 100 == 0:
            avg_loss = total_loss / global_step
            print(f"Step {global_step} | Avg loss: {avg_loss:.4f}")

print("Saving FINAL LoRA adapter...")
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

print("FINAL training completed.")
print("LoRA adapter saved to:", output_dir)
