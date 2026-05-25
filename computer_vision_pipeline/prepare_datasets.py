import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


CLASSES = ["heart", "oblong", "oval", "round", "square"]


def copy_files(file_list, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    for file in file_list:
        shutil.copy(file, target_dir / file.name)


# ---------------------------
# Dataset 1 processing
# ---------------------------
def process_dataset1(raw_path, processed_path):
    train_src = raw_path / "FaceShape Dataset" / "training_set"
    test_src = raw_path / "FaceShape Dataset" / "testing_set"

    for cls in CLASSES:
        cls_cap = cls.capitalize()

        train_images = list((train_src / cls_cap).glob("*"))

        train_split, val_split = train_test_split(
            train_images, test_size=0.2, random_state=42
        )

        copy_files(train_split, processed_path / "train" / cls)
        copy_files(val_split, processed_path / "val" / cls)

        test_images = list((test_src / cls_cap).glob("*"))
        copy_files(test_images, processed_path / "test" / cls)


# ---------------------------
# Dataset 2 processing
# ---------------------------
def process_dataset2(raw_path, processed_path):
    src = raw_path / "faceshape-master" / "published_dataset"

    for cls in CLASSES:
        images = list((src / cls).glob("*"))

        train, temp = train_test_split(images, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)

        copy_files(train, processed_path / "train" / cls)
        copy_files(val, processed_path / "val" / cls)
        copy_files(test, processed_path / "test" / cls)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent

    raw_dataset1 = project_root / "data" / "raw" / "dataset1"
    raw_dataset2 = project_root / "data" / "raw" / "dataset2"

    processed_dataset1 = project_root / "data" / "processed" / "dataset1"
    processed_dataset2 = project_root / "data" / "processed" / "dataset2"

    print("Processing Dataset 1...")
    process_dataset1(raw_dataset1, processed_dataset1)

    print("Processing Dataset 2...")
    process_dataset2(raw_dataset2, processed_dataset2)

    print("Done.")