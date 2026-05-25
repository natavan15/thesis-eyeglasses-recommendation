import os
import base64
import csv
import json
import time
from openai import OpenAI

client = OpenAI()

# Paths
DATASET_PATH = "openai_vision_pipeline/data/github_cropped_75"
OUTPUT_CSV = "openai_vision_pipeline/results/openai_vision_cropped75.csv"
PROMPT_PATH = "openai_vision_pipeline/prompts/face_shape_prompt.txt"

# Full RAW test
MAX_IMAGES = 75

# Delay to avoid rate limit
DELAY_BETWEEN_REQUESTS = 4.0
RETRY_DELAY = 10.0
MAX_RETRIES = 5

valid_labels = {"heart", "oblong", "oval", "round", "square"}

# Load prompt
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    PROMPT = f.read()


def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def clean_json_text(text):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return text


def call_openai_with_retry(base64_image):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
                temperature=0
            )

            return response

        except Exception as e:
            last_error = e
            print(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(RETRY_DELAY)

    raise last_error


results = []
count = 0

for label in sorted(os.listdir(DATASET_PATH)):
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    for img_name in sorted(os.listdir(label_path)):
        if count >= MAX_IMAGES:
            break

        if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        img_path = os.path.join(label_path, img_name)

        print(f"Processing {count + 1}/{MAX_IMAGES}: {img_path}")

        try:
            base64_image = encode_image(img_path)

            response = call_openai_with_retry(base64_image)

            output_text = response.choices[0].message.content.strip()
            clean_text = clean_json_text(output_text)

            try:
                parsed = json.loads(clean_text)

                pred = str(parsed.get("face_shape", "")).lower().strip()
                conf = parsed.get("confidence", "")
                reason = parsed.get("reason", "")

                if pred not in valid_labels:
                    pred = ""

            except Exception:
                pred = ""
                conf = ""
                reason = clean_text

            results.append([
                img_path,
                label,
                pred,
                conf,
                reason,
                clean_text
            ])

        except Exception as e:
            print(f"Final error for image: {img_path}")
            print(e)

            results.append([
                img_path,
                label,
                "",
                "",
                f"ERROR: {e}",
                ""
            ])

        count += 1

        # Important delay to avoid rate limit
        time.sleep(DELAY_BETWEEN_REQUESTS)

    if count >= MAX_IMAGES:
        break


# Save CSV
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_path",
        "true_label",
        "predicted_label",
        "confidence",
        "reason",
        "raw_response"
    ])
    writer.writerows(results)

print(f"DONE. Saved {len(results)} results to: {OUTPUT_CSV}")