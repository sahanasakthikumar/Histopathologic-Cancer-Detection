# Data Preprocessing & Neural Network Training

## Overview
Loads and preprocesses 10,000 histopathology images for cancer detection. Implements parallel data loading and trains CNN models using PyTorch.

## Files
- `data_loader.py` - Parallel image loading (4.98x speedup using multiprocessing)
- `train_pytorch.py` - PyTorch CNN training on CPU and GPU

## Key Features
- Image resizing to 96×96 pixels
- Normalization and data augmentation
- Train/validation/test split (70/15/15)
- Binary classification (cancer vs non-cancer)

## Results
- Dataset: 10,000 images (60.3% class 0, 39.7% class 1)
- Training set: 7,000 samples
- Validation set: 1,500 samples
- Test set: 1,500 samples

## Usage
```bash
python data_loader.py
python train_pytorch.py