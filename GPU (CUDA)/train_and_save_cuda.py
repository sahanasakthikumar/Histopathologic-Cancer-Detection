"""
Fast CUDA training with model saving
Uses the already-optimized CUDA code
"""

import numpy as np
import pickle
import time
from pathlib import Path
import json

# Import the fast CUDA CNN
import sys
sys.path.append(str(Path(__file__).parent))

from cnn_cuda import SimpleCNN_CUDA

def train_and_save_cuda_model(X_train, y_train, X_val, y_val, 
                               epochs=10, batch_size=16):
    """
    Train CUDA model and save weights
    """
    print("\n" + "="*80)
    print("TRAINING GPU MODEL (CUDA)")
    print("="*80)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print("="*80)
    
    # Create model
    model = SimpleCNN_CUDA()
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_times': []
    }
    
    total_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        n_samples = len(X_train)
        indices = np.random.permutation(n_samples)
        
        epoch_loss = 0
        epoch_acc = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices].reshape(-1, 1)
            
            # Forward pass
            predictions = model.forward(X_batch)
            
            # Compute metrics
            loss = model.compute_loss(predictions, y_batch)
            acc = model.compute_accuracy(predictions, y_batch)
            
            epoch_loss += loss
            epoch_acc += acc
            n_batches += 1
            
            if n_batches % 10 == 0:
                print(f"  Batch {n_batches}/{(n_samples + batch_size - 1) // batch_size} - Loss: {loss:.4f}, Acc: {acc:.2f}%", end='\r')
        
        print()  # New line
        
        # Validation
        val_predictions = model.forward(X_val[:batch_size])
        val_targets = y_val[:batch_size].reshape(-1, 1)
        val_loss = model.compute_loss(val_predictions, val_targets)
        val_acc = model.compute_accuracy(val_predictions, val_targets)
        
        epoch_time = time.time() - epoch_start
        
        # Store history
        history['train_loss'].append(float(epoch_loss / n_batches))
        history['train_acc'].append(float(epoch_acc / n_batches))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        history['epoch_times'].append(float(epoch_time))
        
        print(f"\nEpoch {epoch+1}/{epochs} - Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {epoch_loss/n_batches:.4f} | Train Acc: {epoch_acc/n_batches:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    total_time = time.time() - total_start
    
    print("\n" + "="*80)
    print(f"✓ Training completed in {total_time:.2f} seconds")
    print(f"✓ Average time per epoch: {total_time/epochs:.2f} seconds")
    print("="*80)
    
    # Save model weights
    model_data = {
        'conv1_weights': model.conv1.weights,
        'conv1_bias': model.conv1.bias,
        'conv2_weights': model.conv2.weights,
        'conv2_bias': model.conv2.bias,
        'fc1_weights': model.fc1.weights,
        'fc1_bias': model.fc1.bias,
        'fc2_weights': model.fc2.weights,
        'fc2_bias': model.fc2.bias,
        'history': history,
        'model_type': 'cuda'
    }
    
    return model_data, total_time


def main():
    """Main function"""
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "processed"
    MODELS_DIR = BASE_DIR / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("FAST CUDA MODEL TRAINING")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    X_train = np.load(DATA_DIR / "train_images.npy")
    y_train = np.load(DATA_DIR / "train_labels.npy")
    X_val = np.load(DATA_DIR / "val_images.npy")
    y_val = np.load(DATA_DIR / "val_labels.npy")
    
    # Use larger subset
    TRAIN_SIZE = 7000  # Same as OpenMP for fair comparison
    VAL_SIZE = 1500
    
    X_train = X_train[:TRAIN_SIZE]
    y_train = y_train[:TRAIN_SIZE]
    X_val = X_val[:VAL_SIZE]
    y_val = y_val[:VAL_SIZE]
    
    print(f"✓ Using {len(X_train)} train samples")
    print(f"✓ Using {len(X_val)} val samples")
    
    # Train model
    model_data, training_time = train_and_save_cuda_model(
        X_train, y_train, X_val, y_val,
        epochs=20,  # More epochs
        batch_size=16
    )
    
    # Save model
    model_path = MODELS_DIR / "cuda_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✓ Model saved: {model_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'cuda',
        'train_size': TRAIN_SIZE,
        'val_size': VAL_SIZE,
        'epochs': 20,
        'training_time': training_time,
        'final_train_acc': model_data['history']['train_acc'][-1],
        'final_val_acc': model_data['history']['val_acc'][-1]
    }
    
    with open(MODELS_DIR / "cuda_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Metadata saved: {MODELS_DIR / 'cuda_metadata.json'}")
    print(f"\nFinal Results:")
    print(f"  Training Time: {training_time:.2f}s")
    print(f"  Final Val Accuracy: {metadata['final_val_acc']:.2f}%")


if __name__ == "__main__":
    main()