from pathlib import Path
from typing import Tuple

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from computer_vision_pipeline.src.data.transforms import get_train_transforms, get_eval_transforms


class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


def load_imagefolder_dataset(dataset_path: Path, train: bool = True):
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    transform = get_train_transforms() if train else get_eval_transforms()
    dataset = CustomImageFolder(dataset_path, transform=transform)
    return dataset


def create_dataloader(dataset, batch_size: int = 32, shuffle: bool = False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def load_train_val_test_datasets(
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
):
    train_dataset = load_imagefolder_dataset(train_dir, train=True)
    val_dataset = load_imagefolder_dataset(val_dir, train=False)
    test_dataset = load_imagefolder_dataset(test_dir, train=False)

    return train_dataset, val_dataset, test_dataset


def load_train_val_test_dataloaders(
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, val_dataset, test_dataset = load_train_val_test_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
    )

    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader