# Dataset

## MSID — Monkeypox Skin Image Dataset

- **Source**: [Kaggle — Monkeypox Skin Images Dataset (MSID)](https://www.kaggle.com/datasets/dipuiucse/monkeypoxskinimagedataset)
- **Size**: ~770 original images, 4 classes
- **Classes**: Normal, Monkeypox, Chickenpox, Measles
- **Image size**: 224 × 224 (resized during preprocessing)
- **License**: Refer to the dataset's Kaggle page for licence details

## Setup

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/dipuiucse/monkeypoxskinimagedataset)
2. Extract and place the class folders under `data/Monkeypox Skin Image Dataset/`:
   ```
   data/Monkeypox Skin Image Dataset/
   ├── Normal/
   ├── Monkeypox/
   ├── Chickenpox/
   └── Measles/
   ```
3. Run the automated data preparation (balanced augmentation + splitting):
   ```bash
   python train.py --config configs/proposed.yaml --prepare-data --skip-phase1 --device cpu
   ```
   This creates `data/processed/` with `train/`, `val/`, `test/` subdirectories.

4. Alternatively, call the function directly:
   ```python
   from utils.augmentation import organize_balanced_dataset
   organize_balanced_dataset(
       source_path="data/Monkeypox Skin Image Dataset",
       target_path="data/processed",
       target_samples=550,
       train_ratio=0.70,
       val_ratio=0.15,
   )
   ```

## Splits & Augmentation

| Split | Ratio | Augmentation |
|-------|-------|--------------|
| Train | 70%   | Balanced to ~550/class via offline augmentation |
| Val   | 15%   | None (resize + normalise only) |
| Test  | 15%   | None (resize + normalise only) |

**Training augmentations** (Albumentations):
- Horizontal/Vertical flip
- Rotation (±15°)
- Shift-Scale-Rotate
- Brightness/Contrast adjustment
- Hue-Saturation-Value jitter
- Gaussian noise / blur
- CoarseDropout (cutout)
- ImageNet normalisation

## Class Distribution (Original)

| Class | Approx. Count |
|-------|---------------|
| Normal | ~180 |
| Monkeypox | ~230 |
| Chickenpox | ~180 |
| Measles | ~180 |

After balanced augmentation, each class has approximately 550 training images.
