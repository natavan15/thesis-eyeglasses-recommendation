import os
from datasets import Dataset

DATA_PATH = "/workspace/data/FaceShape Dataset/testing_set"

labels = ["Heart", "Oblong", "Oval", "Round", "Square"]
data = []

for label in labels:
    folder = os.path.join(DATA_PATH, label)

    for img_name in os.listdir(folder):
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            data.append({
                "image_path": os.path.join(folder, img_name),
                "label": label.lower()
            })

dataset = Dataset.from_list(data)
dataset.save_to_disk("/workspace/kaggle_validation_dataset")

print("Validation dataset prepared:", len(dataset))
print(dataset[0])
