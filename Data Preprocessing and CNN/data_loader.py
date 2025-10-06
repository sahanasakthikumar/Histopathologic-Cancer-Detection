"""
Histopathology Image Data Loader with Parallel Processing
Loads 10,000 .tif images and preprocesses them for CNN training
"""

import pandas as pd
import numpy as np
from PIL import Image
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

class HistopathDataLoader:
    def __init__(self, excel_path, image_folder, img_size=(96, 96)):
        """
        Initialize data loader
        
        Args:
            excel_path: Path to Excel file with id and label columns
            image_folder: Path to folder containing .tif images
            img_size: Target size for resizing images (height, width)
        """
        self.excel_path = excel_path
        self.image_folder = Path(image_folder)
        self.img_size = img_size
        
    def load_metadata(self):
        """Load image IDs and labels from Excel file"""
        print(f"Loading metadata from: {self.excel_path}")
        df = pd.read_excel(self.excel_path)
        print(f"✓ Loaded {len(df)} samples")
        print(f"✓ Columns: {df.columns.tolist()}")
        
        # Check class distribution
        print("\nClass Distribution:")
        print(df['label'].value_counts().sort_index())
        
        return df
    
    def load_single_image(self, args):
        """
        Load and preprocess a single image
        
        Args:
            args: Tuple of (img_id, label)
        
        Returns:
            Tuple of (img_id, img_array, label) or (img_id, None, label) if failed
        """
        img_id, label = args
        img_path = self.image_folder / f"{img_id}.tif"
        
        if not img_path.exists():
            return (img_id, None, label)
        
        try:
            # Load image
            img = Image.open(img_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to target size
            img = img.resize(self.img_size, Image.LANCZOS)
            
            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Transpose to channel-first format: (H, W, C) -> (C, H, W)
            img_array = np.transpose(img_array, (2, 0, 1))
            
            return (img_id, img_array, label)
        
        except Exception as e:
            print(f"Error loading {img_id}: {str(e)}")
            return (img_id, None, label)
    
    def load_dataset_sequential(self, df):
        """
        Load images sequentially (baseline - slower)
        
        Args:
            df: DataFrame with 'id' and 'label' columns
        
        Returns:
            Tuple of (images, labels, valid_ids)
        """
        print("\n" + "="*60)
        print("SEQUENTIAL LOADING (Baseline)")
        print("="*60)
        
        images = []
        labels = []
        valid_ids = []
        
        start_time = time.time()
        
        args_list = [(row['id'], row['label']) for _, row in df.iterrows()]
        
        for args in tqdm(args_list, desc="Loading images"):
            img_id, img_array, label = self.load_single_image(args)
            
            if img_array is not None:
                images.append(img_array)
                labels.append(label)
                valid_ids.append(img_id)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"\n✓ Sequential loading completed in {elapsed:.2f} seconds")
        print(f"✓ Successfully loaded: {len(images)}/{len(df)} images")
        print(f"✓ Loading speed: {len(images)/elapsed:.2f} images/second")
        
        return np.array(images), np.array(labels), valid_ids, elapsed
    
    def load_dataset_parallel(self, df, n_workers=None):
        """
        Load images in parallel using multiprocessing (faster)
        
        Args:
            df: DataFrame with 'id' and 'label' columns
            n_workers: Number of parallel workers (default: CPU count)
        
        Returns:
            Tuple of (images, labels, valid_ids)
        """
        if n_workers is None:
            n_workers = cpu_count()
        
        print("\n" + "="*60)
        print(f"PARALLEL LOADING (Using {n_workers} CPU cores)")
        print("="*60)
        
        # Prepare arguments for parallel processing
        args_list = [(row['id'], row['label']) for _, row in df.iterrows()]
        
        start_time = time.time()
        
        # Parallel processing with progress bar
        with Pool(processes=n_workers) as pool:
            results = list(tqdm(
                pool.imap(self.load_single_image, args_list),
                total=len(args_list),
                desc="Loading images"
            ))
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Filter valid results
        images = []
        labels = []
        valid_ids = []
        
        for img_id, img_array, label in results:
            if img_array is not None:
                images.append(img_array)
                labels.append(label)
                valid_ids.append(img_id)
        
        print(f"\n✓ Parallel loading completed in {elapsed:.2f} seconds")
        print(f"✓ Successfully loaded: {len(images)}/{len(df)} images")
        print(f"✓ Loading speed: {len(images)/elapsed:.2f} images/second")
        
        return np.array(images), np.array(labels), valid_ids, elapsed
    
    def split_dataset(self, images, labels, train_ratio=0.7, val_ratio=0.15):
        """
        Split dataset into train, validation, and test sets
        
        Args:
            images: Array of images
            labels: Array of labels
            train_ratio: Proportion for training (default: 70%)
            val_ratio: Proportion for validation (default: 15%)
            
        Returns:
            Dictionary with train, val, test splits
        """
        n_samples = len(images)
        indices = np.random.permutation(n_samples)
        
        train_end = int(train_ratio * n_samples)
        val_end = train_end + int(val_ratio * n_samples)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        print("\n" + "="*60)
        print("DATASET SPLIT")
        print("="*60)
        print(f"Train set: {len(train_idx)} samples ({len(train_idx)/n_samples*100:.1f}%)")
        print(f"Validation set: {len(val_idx)} samples ({len(val_idx)/n_samples*100:.1f}%)")
        print(f"Test set: {len(test_idx)} samples ({len(test_idx)/n_samples*100:.1f}%)")
        
        return {
            'train': (images[train_idx], labels[train_idx]),
            'val': (images[val_idx], labels[val_idx]),
            'test': (images[test_idx], labels[test_idx])
        }
    
    def save_processed_data(self, splits, output_dir):
        """
        Save processed data as .npy files
        
        Args:
            splits: Dictionary with train/val/test splits
            output_dir: Directory to save processed data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("SAVING PROCESSED DATA")
        print("="*60)
        
        for split_name, (images, labels) in splits.items():
            img_path = output_path / f"{split_name}_images.npy"
            label_path = output_path / f"{split_name}_labels.npy"
            
            np.save(img_path, images)
            np.save(label_path, labels)
            
            print(f"✓ Saved {split_name}_images.npy: {images.shape}")
            print(f"✓ Saved {split_name}_labels.npy: {labels.shape}")
        
        print(f"\n✓ All data saved to: {output_path}")


def main():
    """Main function to run data preprocessing"""
    
    # Configuration
    BASE_DIR = Path(__file__).parent.parent.parent  # Go up to project root
    EXCEL_PATH = BASE_DIR / "data" / "10000_small.xlsx"
    IMAGE_FOLDER = BASE_DIR / "data" / "images"
    OUTPUT_DIR = BASE_DIR / "data" / "processed"
    IMG_SIZE = (96, 96)  # Resize images to 96x96
    
    print("="*60)
    print("HISTOPATHOLOGY DATA PREPROCESSING")
    print("="*60)
    print(f"Excel file: {EXCEL_PATH}")
    print(f"Image folder: {IMAGE_FOLDER}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Target image size: {IMG_SIZE}")
    
    # Initialize loader
    loader = HistopathDataLoader(EXCEL_PATH, IMAGE_FOLDER, IMG_SIZE)
    
    # Load metadata
    df = loader.load_metadata()
    
    # Option 1: Sequential loading (for comparison)
    print("\n" + "="*60)
    print("TESTING SEQUENTIAL VS PARALLEL LOADING")
    print("="*60)
    
    # Test on a small subset first (100 images)
    test_df = df.head(100)
    print(f"\nTesting with {len(test_df)} images...")
    
    _, _, _, seq_time = loader.load_dataset_sequential(test_df)
    _, _, _, par_time = loader.load_dataset_parallel(test_df, n_workers=4)
    
    speedup = seq_time / par_time
    print(f"\n🚀 SPEEDUP: {speedup:.2f}x faster with parallel loading!")
    
    # Now load full dataset with parallel loading
    print("\n" + "="*60)
    print("LOADING FULL DATASET (10,000 images)")
    print("="*60)
    
    images, labels, valid_ids, _ = loader.load_dataset_parallel(df, n_workers=4)
    
    # Split dataset
    splits = loader.split_dataset(images, labels)
    
    # Save processed data
    loader.save_processed_data(splits, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("✓ DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()