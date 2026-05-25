import os
from datasets import Dataset

DATA_PATH = "/workspace/data/cnn_cropped_faces/processed_faces/dataset1/train"

data = []

for label in os.listdir(DATA_PATH):
    label_path = os.path.join(DATA_PATH, label)
    if not os.path.isdir(label_path):
        continue

    for img in os.listdir(label_path):
        if img.lower().endswith((".jpg", ".jpeg", ".png")):
            data.append({
                "image_path": os.path.join(label_path, img),
                "label": label
            })

dataset = Dataset.from_list(data)
dataset.save_to_disk("/workspace/face_shape_dataset_cropped")

print("Cropped training dataset:", len(dataset))
