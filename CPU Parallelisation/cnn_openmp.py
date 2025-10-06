"""
Python wrapper for OpenMP-accelerated CNN operations
"""

import numpy as np
import ctypes
from pathlib import Path
import time
import matplotlib.pyplot as plt

# Load the DLL
dll_path = Path(__file__).parent / "conv_openmp.dll"
if not dll_path.exists():
    raise FileNotFoundError(f"DLL not found: {dll_path}\nPlease compile conv_openmp.cpp first!")

lib = ctypes.CDLL(str(dll_path))

# Define function signatures
lib.im2col_openmp.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]

lib.matmul_openmp.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int
]

lib.relu_openmp.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int
]

lib.sigmoid_openmp.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int
]

lib.maxpool_openmp.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int
]

lib.get_num_threads.restype = ctypes.c_int
lib.set_num_threads.argtypes = [ctypes.c_int]


class ConvLayerOpenMP:
    """Convolution layer using OpenMP acceleration"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.bias = np.zeros(out_channels, dtype=np.float32)
        
        self.weights_col = self.weights.reshape(out_channels, -1).astype(np.float32)
    
    def forward(self, x):
        """Forward pass using OpenMP"""
        batch_size, _, h, w = x.shape
        
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Allocate column matrix
        col_size = batch_size * out_h * out_w * self.in_channels * self.kernel_size * self.kernel_size
        col = np.zeros(col_size, dtype=np.float32)
        
        # Call OpenMP im2col
        x_flat = x.flatten().astype(np.float32)
        lib.im2col_openmp(
            x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            col.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch_size, self.in_channels, h, w,
            self.kernel_size, self.stride, self.padding, out_h, out_w
        )
        
        # Reshape col for matrix multiplication
        col = col.reshape(batch_size * out_h * out_w, -1)
        
        # Matrix multiplication using OpenMP
        output = np.zeros((batch_size * out_h * out_w, self.out_channels), dtype=np.float32)
        lib.matmul_openmp(
            col.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.weights_col.T.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch_size * out_h * out_w,
            self.out_channels,
            col.shape[1]
        )
        
        # Add bias
        output = output + self.bias
        
        # Reshape to output format
        output = output.reshape(batch_size, out_h, out_w, self.out_channels)
        output = np.transpose(output, (0, 3, 1, 2))
        
        return output


class MaxPoolLayerOpenMP:
    """Max pooling using OpenMP"""
    
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, x):
        """Forward pass using OpenMP"""
        batch_size, channels, h, w = x.shape
        
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_h, out_w), dtype=np.float32)
        
        x_flat = x.flatten().astype(np.float32)
        output_flat = output.flatten()
        
        lib.maxpool_openmp(
            x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch_size, channels, h, w,
            self.pool_size, self.stride
        )
        
        return output_flat.reshape(batch_size, channels, out_h, out_w)


class FullyConnectedLayerOpenMP:
    """FC layer with OpenMP matrix multiplication"""
    
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        self.weights = (np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)).astype(np.float32)
        self.bias = np.zeros(output_size, dtype=np.float32)
    
    def forward(self, x):
        """Forward pass using OpenMP"""
        batch_size = x.shape[0]
        output = np.zeros((batch_size, self.output_size), dtype=np.float32)
        
        lib.matmul_openmp(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch_size, self.output_size, self.input_size
        )
        
        return output + self.bias


class SimpleCNN_OpenMP:
    """CNN using OpenMP acceleration"""
    
    def __init__(self, num_threads=4):
        lib.set_num_threads(num_threads)
        print(f"OpenMP threads: {lib.get_num_threads()}")
        
        self.conv1 = ConvLayerOpenMP(3, 16, kernel_size=3, padding=1)
        self.pool1 = MaxPoolLayerOpenMP(pool_size=2, stride=2)
        
        self.conv2 = ConvLayerOpenMP(16, 32, kernel_size=3, padding=1)
        self.pool2 = MaxPoolLayerOpenMP(pool_size=2, stride=2)
        
        self.flat_size = 32 * 24 * 24
        self.fc1 = FullyConnectedLayerOpenMP(self.flat_size, 128)
        self.fc2 = FullyConnectedLayerOpenMP(128, 1)
    
    def forward(self, x):
        """Forward pass"""
        x = x.astype(np.float32)
        
        # Conv1 -> ReLU -> Pool1
        out = self.conv1.forward(x)
        out = np.maximum(0, out)  # ReLU
        out = self.pool1.forward(out)
        
        # Conv2 -> ReLU -> Pool2
        out = self.conv2.forward(out)
        out = np.maximum(0, out)  # ReLU
        out = self.pool2.forward(out)
        
        # Flatten
        batch_size = out.shape[0]
        out = out.reshape(batch_size, -1)
        
        # FC1 -> ReLU
        out = self.fc1.forward(out)
        out = np.maximum(0, out)  # ReLU
        
        # FC2 -> Sigmoid
        out = self.fc2.forward(out)
        
        # Sigmoid activation
        out_sigmoid = np.zeros_like(out)
        lib.sigmoid_openmp(
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_sigmoid.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.size
        )
        
        return out_sigmoid
    
    def compute_loss(self, predictions, targets):
        """Binary cross-entropy loss"""
        epsilon = 1e-7
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        return loss
    
    def compute_accuracy(self, predictions, targets):
        """Compute accuracy"""
        pred_labels = (predictions >= 0.5).astype(int).flatten()
        return np.mean(pred_labels == targets) * 100


class Trainer:
    """Training loop for OpenMP CNN"""
    
    def __init__(self, model):
        self.model = model
        self.train_times = []
    
    def train_epoch(self, X_train, y_train, batch_size=32):
        """Train one epoch"""
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
            predictions = self.model.forward(X_batch)
            
            # Compute metrics
            loss = self.model.compute_loss(predictions, y_batch)
            acc = self.model.compute_accuracy(predictions, y_batch)
            
            epoch_loss += loss
            epoch_acc += acc
            n_batches += 1
        
        return epoch_loss / n_batches, epoch_acc / n_batches
    
    def evaluate(self, X_val, y_val, batch_size=32):
        """Evaluate"""
        predictions = self.model.forward(X_val[:batch_size])
        y_batch = y_val[:batch_size].reshape(-1, 1)
        
        loss = self.model.compute_loss(predictions, y_batch)
        acc = self.model.compute_accuracy(predictions, y_batch)
        
        return loss, acc
    
    def train(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=32):
        """Training loop"""
        print("\n" + "="*60)
        print("OPENMP-ACCELERATED CNN TRAINING")
        print("="*60)
        
        total_start = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            train_loss, train_acc = self.train_epoch(X_train, y_train, batch_size)
            val_loss, val_acc = self.evaluate(X_val, y_val, batch_size)
            
            epoch_time = time.time() - epoch_start
            self.train_times.append(epoch_time)
            
            print(f"\nEpoch {epoch+1}/{epochs} - Time: {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        total_time = time.time() - total_start
        
        print("\n" + "="*60)
        print(f"✓ Training completed in {total_time:.2f} seconds")
        print(f"✓ Average time per epoch: {total_time/epochs:.2f} seconds")
        print("="*60)
        
        return total_time


def main():
    """Main function"""
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "processed"
    RESULTS_DIR = BASE_DIR / "results" / "cpu_benchmarks"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    X_train = np.load(DATA_DIR / "train_images.npy")
    y_train = np.load(DATA_DIR / "train_labels.npy")
    X_val = np.load(DATA_DIR / "val_images.npy")
    y_val = np.load(DATA_DIR / "val_labels.npy")
    
    # Use subset
    X_train = X_train[:1000]
    y_train = y_train[:1000]
    X_val = X_val[:200]
    y_val = y_val[:200]
    
    print(f"✓ Using: {len(X_train)} train, {len(X_val)} val samples")
    
    # Create model with 4 threads
    model = SimpleCNN_OpenMP(num_threads=4)
    trainer = Trainer(model)
    
    # Train
    training_time = trainer.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=16)
    
    # Save results
    with open(RESULTS_DIR / "openmp_timing.txt", "w") as f:
        f.write(f"OpenMP Training Time: {training_time:.2f} seconds\n")
        f.write(f"Average per epoch: {training_time/5:.2f} seconds\n")
    
    print("\n✓ OPENMP TRAINING COMPLETED!")


if __name__ == "__main__":
    main()