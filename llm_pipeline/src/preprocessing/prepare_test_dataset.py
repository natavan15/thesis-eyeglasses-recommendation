import os
from datasets import Dataset

DATA_PATH = "/workspace/data/github_test/faceshape-master/published_dataset"

labels = ["heart", "oblong", "oval", "round", "square"]
data = []

for label in labels:
    folder = os.path.join(DATA_PATH, label)

    for img_name in os.listdir(folder):
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            data.append({
                "image_path": os.path.join(folder, img_name),
                "label": label
            })

dataset = Dataset.from_list(data)
dataset.save_to_disk("/workspace/github_test_dataset")

print("Test dataset prepared:", len(dataset))
