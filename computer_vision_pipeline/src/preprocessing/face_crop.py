from pathlib import Path
import cv2
from mtcnn import MTCNN


detector = MTCNN()
MAX_IMAGE_SIDE = 1200  # reduce very large images before detection


def safe_resize_for_detection(img):
    h, w = img.shape[:2]
    max_side = max(h, w)

    if max_side <= MAX_IMAGE_SIDE:
        return img, 1.0

    scale = MAX_IMAGE_SIDE / max_side
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))
    return resized, scale


def crop_face(image_path: Path, save_path: Path) -> bool:
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Could not read image: {image_path}")
            return False

        # Resize large images before detection to avoid memory issues
        img_small, scale = safe_resize_for_detection(img)
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        results = detector.detect_faces(img_rgb)

        if len(results) == 0:
            print(f"No face detected: {image_path}")
            return False

        x, y, w, h = results[0]["box"]

        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)

        # Map crop box back to original image size if resized
        if scale != 1.0:
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)

        x2 = min(img.shape[1], x + w)
        y2 = min(img.shape[0], y + h)

        face = img[y:y2, x:x2]

        if face.size == 0:
            print(f"Empty crop: {image_path}")
            return False

        face = cv2.resize(face, (224, 224))

        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), face)
        return True

    except Exception as e:
        print(f"Skipping {image_path} بسبب error: {e}")
        return False


def process_split(input_dir: Path, output_dir: Path):
    total = 0
    success = 0

    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue

        for img_path in class_dir.glob("*"):
            if not img_path.is_file():
                continue

            total += 1
            save_path = output_dir / class_dir.name / img_path.name

            ok = crop_face(img_path, save_path)
            if ok:
                success += 1

    print(f"Processed {input_dir}")
    print(f"Successfully cropped {success}/{total} images\n")


def main():
    project_root = Path(__file__).resolve().parents[2]

    splits = ["train", "val", "test"]
    datasets = ["dataset1", "dataset2"]

    for dataset in datasets:
        for split in splits:
            input_dir = project_root / "data" / "processed" / dataset / split
            output_dir = project_root / "data" / "processed_faces" / dataset / split

            if input_dir.exists():
                process_split(input_dir, output_dir)
            else:
                print(f"Skipping missing folder: {input_dir}")


if __name__ == "__main__":
    main()