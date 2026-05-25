from pathlib import Path

from computer_vision_pipeline.src.data.dataset import load_imagefolder_dataset

# change this only when you already place real data there
sample_path = Path("data/dataset1")

print("Testing dataset path:", sample_path.resolve())

if sample_path.exists():
    print("Path exists.")
else:
    print("Path does not exist yet. This is okay for now.")