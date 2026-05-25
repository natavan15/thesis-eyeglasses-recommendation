from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_path = r"D:\Thesis\thesis-eyeglasses\llm_pipeline\models\gemma4"

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

print("Model loaded! You can start asking questions.")
print("Type 'exit' to quit.\n")

while True:
    user_question = input("You: ")

    if user_question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_question}
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
            max_new_tokens=30,
            do_sample=False
        )

    response = processor.decode(output[0], skip_special_tokens=True)

    print("\nModel:", response, "\n")