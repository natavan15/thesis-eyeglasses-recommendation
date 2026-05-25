from datasets import load_from_disk, Dataset
from PIL import Image
from tqdm import tqdm

ds = load_from_disk("/workspace/face_shape_dataset")

good = []
bad = []

for ex in tqdm(ds):
    try:
        img = Image.open(ex["image_path"])
        img.load()
        good.append(ex)
    except Exception as e:
        bad.append((ex["image_path"], str(e)))

print("Good images:", len(good))
print("Bad images:", len(bad))

for path, err in bad:
    print("BAD:", path, "|", err)

clean_ds = Dataset.from_list(good)
clean_ds.save_to_disk("/workspace/face_shape_dataset_clean")

print("Clean dataset saved to /workspace/face_shape_dataset_clean")
