"""
Data augmentation pipelines and dataset classes for XDenseQNet.

Uses Albumentations for training augmentation and provides balanced
data augmentation to equalise class counts.
"""

import os
import random
import shutil
from typing import Dict, List, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ImageNet normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ["Normal", "Monkeypox", "Chickenpox", "Measles"]
CLASS_DIRS = ["normal", "monkeypox", "chickenpox", "measles"]


# ---------------------------------------------------------------------------
#  Augmentation transforms
# ---------------------------------------------------------------------------


def get_transforms(
    image_size: Tuple[int, int] = (224, 224),
) -> Tuple[A.Compose, A.Compose, A.Compose]:
    """Return (train_transform, val_transform, test_transform)."""

    train_transform = A.Compose(
        [
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3
            ),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                ],
                p=0.3,
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=16,
                max_width=16,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.3,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    val_test_transform = A.Compose(
        [
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    return train_transform, val_test_transform, val_test_transform


# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------


class MultiClassDataset(Dataset):
    """Skin-image classification dataset (MSID).

    Expects the directory layout::

        data_dir/
            normal/
            monkeypox/
            chickenpox/
            measles/
    """

    def __init__(
        self,
        data_dir: str,
        transform=None,
        image_size: Tuple[int, int] = (224, 224),
        split_name: str = "train",
    ):
        self.transform = transform
        self.image_size = image_size
        self.images: List[str] = []
        self.labels: List[int] = []
        self.class_counts: Dict[int, int] = {}

        for class_idx, class_name in enumerate(CLASS_DIRS):
            class_path = os.path.join(data_dir, class_name)
            if os.path.exists(class_path):
                imgs = [
                    os.path.join(class_path, f)
                    for f in os.listdir(class_path)
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
                ]
                self.images.extend(imgs)
                self.labels.extend([class_idx] * len(imgs))
                self.class_counts[class_idx] = len(imgs)

        print(f"  {split_name.upper()}: {len(self.images)} images")
        for idx, name in enumerate(CLASS_NAMES):
            print(f"    {name}: {self.class_counts.get(idx, 0)}")

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for balanced training."""
        total = sum(self.class_counts.values())
        n = len(self.class_counts)
        return torch.FloatTensor(
            [total / (n * max(self.class_counts.get(i, 1), 1)) for i in range(n)]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = image.resize(self.image_size, Image.LANCZOS)
            image = np.array(image)
        except Exception:
            image = np.zeros(
                (self.image_size[0], self.image_size[1], 3), dtype=np.uint8
            )
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label


# ---------------------------------------------------------------------------
#  Data loaders
# ---------------------------------------------------------------------------


def create_dataloaders(
    processed_dir: str,
    batch_size: int = 16,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 0,
) -> Dict:
    """Create train / val / test DataLoaders from the processed dataset."""

    train_tf, val_tf, test_tf = get_transforms(image_size)

    train_ds = MultiClassDataset(
        os.path.join(processed_dir, "train"), train_tf, image_size, "train"
    )
    val_ds = MultiClassDataset(
        os.path.join(processed_dir, "val"), val_tf, image_size, "val"
    )
    test_ds = MultiClassDataset(
        os.path.join(processed_dir, "test"), test_tf, image_size, "test"
    )

    kw = dict(num_workers=num_workers, pin_memory=True)
    return {
        "train_loader": DataLoader(
            train_ds, batch_size, shuffle=True, drop_last=True, **kw
        ),
        "val_loader": DataLoader(val_ds, batch_size, shuffle=False, **kw),
        "test_loader": DataLoader(test_ds, batch_size, shuffle=False, **kw),
        "train_dataset": train_ds,
        "val_dataset": val_ds,
        "test_dataset": test_ds,
    }


# ---------------------------------------------------------------------------
#  Balanced augmentation (offline, used during dataset preparation)
# ---------------------------------------------------------------------------


class BalancedDataBalancer:
    """Generate augmented images to balance all classes to a target count."""

    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        self.augmentation_transform = A.Compose(
            [
                A.Resize(image_size[0], image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=20, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3
                ),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                    ],
                    p=0.3,
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=16,
                    max_width=16,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.2,
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )

    def generate_augmented_images(
        self, image_path: str, num_augmentations: int
    ) -> List[Image.Image]:
        """Generate ``num_augmentations`` augmented PIL images from a source."""
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize(self.image_size, Image.LANCZOS)
            image_np = np.array(image)

            results = []
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

            for _ in range(num_augmentations):
                aug = self.augmentation_transform(image=image_np)["image"]
                img_denorm = torch.clamp(aug * std + mean, 0, 1)
                img_np = img_denorm.permute(1, 2, 0).numpy()
                results.append(Image.fromarray((img_np * 255).astype(np.uint8)))
            return results
        except Exception:
            return []


def organize_balanced_dataset(
    source_path: str,
    target_path: str,
    target_samples: int = 550,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    image_size: Tuple[int, int] = (224, 224),
) -> bool:
    """Organise raw dataset into train/val/test splits with balanced augmentation.

    Args:
        source_path:    Path to the raw dataset with class subfolders.
        target_path:    Where to write the processed splits.
        target_samples: Target number of training images per class.
        train_ratio:    Fraction of images for training.
        val_ratio:      Fraction of images for validation.
        image_size:     Target image dimensions.

    Returns:
        ``True`` on success.
    """
    print(f"\nOrganising dataset with balanced augmentation")
    print("=" * 70)

    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.makedirs(target_path, exist_ok=True)

    for split in ("train", "val", "test"):
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(target_path, split, cls.lower()), exist_ok=True)

    # Collect per-class images
    all_class_images: Dict[str, List[str]] = {}
    for class_name in CLASS_NAMES:
        for folder in os.listdir(source_path):
            fp = os.path.join(source_path, folder)
            if os.path.isdir(fp) and class_name.lower() in folder.lower():
                imgs = [
                    os.path.join(fp, f)
                    for f in os.listdir(fp)
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
                ]
                all_class_images[class_name] = imgs
                print(f"  {class_name}: {len(imgs)} original images")
                break

    # Split and copy
    split_counts: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    for class_name, images in all_class_images.items():
        random.shuffle(images)
        total = len(images)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        splits = {
            "train": images[:train_size],
            "val": images[train_size : train_size + val_size],
            "test": images[train_size + val_size :],
        }
        for split, split_imgs in splits.items():
            dst = os.path.join(target_path, split, class_name.lower())
            for i, img_path in enumerate(split_imgs):
                shutil.copy2(
                    img_path,
                    os.path.join(dst, f"{class_name.lower()}_{split}_{i:04d}.jpg"),
                )
            split_counts[split][class_name] = len(split_imgs)

    # Balanced augmentation for training
    print(f"\nAugmenting training data to {target_samples} per class...")
    balancer = BalancedDataBalancer(image_size)
    for class_name in CLASS_NAMES:
        current = split_counts["train"][class_name]
        needed = target_samples - current
        target_dir = os.path.join(target_path, "train", class_name.lower())
        if needed > 0:
            orig_files = [
                f
                for f in os.listdir(target_dir)
                if f.endswith((".jpg", ".jpeg", ".png"))
            ][:15]
            total_gen = 0
            per_image = max(1, needed // len(orig_files))
            for idx, img_file in enumerate(orig_files):
                if total_gen >= needed:
                    break
                aug_imgs = balancer.generate_augmented_images(
                    os.path.join(target_dir, img_file), per_image
                )
                for j, aug_img in enumerate(aug_imgs):
                    if total_gen >= needed:
                        break
                    aug_img.save(
                        os.path.join(
                            target_dir,
                            f"{class_name.lower()}_aug_{idx:04d}_{j:02d}.jpg",
                        ),
                        "JPEG",
                        quality=95,
                    )
                    total_gen += 1
            print(f"  {class_name}: {current} -> {current + total_gen} (+{total_gen})")
        else:
            print(f"  {class_name}: {current} (already at target)")

    print("\nDataset organisation complete!")
    return True
