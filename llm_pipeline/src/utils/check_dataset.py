import os

dataset_path = r"D:\Thesis\thesis-eyeglasses\llm_pipeline\data\github_test_dataset"

classes = ["heart", "oblong", "oval", "round", "square"]

print("Checking dataset...\n")

total_images = 0

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)

    if not os.path.exists(class_path):
        print(f"{class_name}: Folder NOT found")
        continue

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    count = len(images)
    total_images += count

    print(f"{class_name}: {count} images")

print(f"\nTotal images: {total_images}")