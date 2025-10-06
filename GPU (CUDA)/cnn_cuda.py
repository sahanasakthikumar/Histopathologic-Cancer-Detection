"""
Python wrapper for CUDA-accelerated CNN
"""

import numpy as np
import ctypes
from pathlib import Path
import time

# Load CUDA DLL
dll_path = Path(__file__).parent / "conv_cuda.dll"
if not dll_path.exists():
    raise FileNotFoundError(f"CUDA DLL not found: {dll_path}\nPlease compile conv_cuda.cu first!")

cuda_lib = ctypes.CDLL(str(dll_path))

# Define function signatures
cuda_lib.cuda_malloc.restype = ctypes.c_void_p
cuda_lib.cuda_malloc.argtypes = [ctypes.c_size_t]

cuda_lib.cuda_free.argtypes = [ctypes.c_void_p]

cuda_lib.cuda_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
cuda_lib.cuda_memcpy_d2h.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_size_t]

cuda_lib.im2col_cuda.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]

cuda_lib.matmul_cuda.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

cuda_lib.relu_cuda.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
cuda_lib.sigmoid_cuda.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
cuda_lib.maxpool_cuda.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int
]


class ConvLayerCUDA:
    """Convolution layer using CUDA"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights on CPU
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.bias = np.zeros(out_channels, dtype=np.float32)
        
        self.weights_col = self.weights.reshape(out_channels, -1).astype(np.float32)
        
        # Allocate GPU memory for weights
        weight_size = self.weights_col.size
        self.d_weights = cuda_lib.cuda_malloc(weight_size)
        cuda_lib.cuda_memcpy_h2d(
            self.d_weights,
            self.weights_col.T.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            weight_size
        )
    
    def forward(self, x, d_input):
        """Forward pass on GPU"""
        batch_size, _, h, w = x.shape
        
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Allocate GPU memory for column matrix
        col_size = batch_size * out_h * out_w * self.in_channels * self.kernel_size * self.kernel_size
        d_col = cuda_lib.cuda_malloc(col_size)
        
        # Call CUDA im2col
        cuda_lib.im2col_cuda(
            d_input, d_col,
            batch_size, self.in_channels, h, w,
            self.kernel_size, self.stride, self.padding, out_h, out_w
        )
        
        # Matrix multiplication on GPU
        output_size = batch_size * out_h * out_w * self.out_channels
        d_output = cuda_lib.cuda_malloc(output_size)
        
        cuda_lib.matmul_cuda(
            d_col, self.d_weights, d_output,
            batch_size * out_h * out_w,
            self.out_channels,
            self.in_channels * self.kernel_size * self.kernel_size
        )
        
        # Copy result back to CPU
        output = np.zeros(output_size, dtype=np.float32)
        cuda_lib.cuda_memcpy_d2h(
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            d_output,
            output_size
        )
        
        # Free temporary GPU memory
        cuda_lib.cuda_free(d_col)
        
        # Add bias and reshape
        output = output.reshape(batch_size, out_h, out_w, self.out_channels)
        output = np.transpose(output, (0, 3, 1, 2))
        output = output + self.bias.reshape(1, -1, 1, 1)
        
        return output, d_output
    
    def __del__(self):
        """Free GPU memory"""
        if hasattr(self, 'd_weights'):
            cuda_lib.cuda_free(self.d_weights)


class MaxPoolLayerCUDA:
    """Max pooling using CUDA"""
    
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, x, d_input):
        """Forward pass on GPU"""
        batch_size, channels, h, w = x.shape
        
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        
        output_size = batch_size * channels * out_h * out_w
        d_output = cuda_lib.cuda_malloc(output_size)
        
        cuda_lib.maxpool_cuda(
            d_input, d_output,
            batch_size, channels, h, w,
            self.pool_size, self.stride
        )
        
        # Copy back to CPU
        output = np.zeros(output_size, dtype=np.float32)
        cuda_lib.cuda_memcpy_d2h(
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            d_output,
            output_size
        )
        
        return output.reshape(batch_size, channels, out_h, out_w), d_output


class FullyConnectedLayerCUDA:
    """FC layer using CUDA"""
    
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        self.weights = (np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)).astype(np.float32)
        self.bias = np.zeros(output_size, dtype=np.float32)
        
        # Allocate GPU memory for weights
        weight_size = self.weights.size
        self.d_weights = cuda_lib.cuda_malloc(weight_size)
        cuda_lib.cuda_memcpy_h2d(
            self.d_weights,
            self.weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            weight_size
        )
    
    def forward(self, x, d_input):
        """Forward pass on GPU"""
        batch_size = x.shape[0]
        output_size = batch_size * self.output_size
        d_output = cuda_lib.cuda_malloc(output_size)
        
        cuda_lib.matmul_cuda(
            d_input, self.d_weights, d_output,
            batch_size, self.output_size, self.input_size
        )
        
        # Copy back and add bias
        output = np.zeros(output_size, dtype=np.float32)
        cuda_lib.cuda_memcpy_d2h(
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            d_output,
            output_size
        )
        
        output = output.reshape(batch_size, self.output_size)
        output = output + self.bias
        
        return output, d_output
    
    def __del__(self):
        """Free GPU memory"""
        if hasattr(self, 'd_weights'):
            cuda_lib.cuda_free(self.d_weights)


class SimpleCNN_CUDA:
    """CNN using CUDA acceleration"""
    
    def __init__(self):
        print("Initializing CUDA CNN...")
        
        self.conv1 = ConvLayerCUDA(3, 16, kernel_size=3, padding=1)
        self.pool1 = MaxPoolLayerCUDA(pool_size=2, stride=2)
        
        self.conv2 = ConvLayerCUDA(16, 32, kernel_size=3, padding=1)
        self.pool2 = MaxPoolLayerCUDA(pool_size=2, stride=2)
        
        self.flat_size = 32 * 24 * 24
        self.fc1 = FullyConnectedLayerCUDA(self.flat_size, 128)
        self.fc2 = FullyConnectedLayerCUDA(128, 1)
        
        print("✓ CUDA CNN initialized")
    
    def forward(self, x):
        """Forward pass using GPU"""
        x = x.astype(np.float32)
        batch_size = x.shape[0]
        
        # Upload input to GPU
        input_size = x.size
        d_input = cuda_lib.cuda_malloc(input_size)
        cuda_lib.cuda_memcpy_h2d(
            d_input,
            x.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            input_size
        )
        
        # Conv1 -> ReLU -> Pool1
        out, d_out = self.conv1.forward(x, d_input)
        cuda_lib.cuda_free(d_input)
        
        # ReLU on GPU
        out = np.maximum(0, out)
        d_input = cuda_lib.cuda_malloc(out.size)
        cuda_lib.cuda_memcpy_h2d(
            d_input,
            out.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.size
        )
        cuda_lib.cuda_free(d_out)
        
        out, d_out = self.pool1.forward(out, d_input)
        cuda_lib.cuda_free(d_input)
        
        # Conv2 -> ReLU -> Pool2
        d_input = cuda_lib.cuda_malloc(out.size)
        cuda_lib.cuda_memcpy_h2d(
            d_input,
            out.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.size
        )
        cuda_lib.cuda_free(d_out)
        
        out, d_out = self.conv2.forward(out, d_input)
        cuda_lib.cuda_free(d_input)
        
        out = np.maximum(0, out)
        d_input = cuda_lib.cuda_malloc(out.size)
        cuda_lib.cuda_memcpy_h2d(
            d_input,
            out.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.size
        )
        cuda_lib.cuda_free(d_out)
        
        out, d_out = self.pool2.forward(out, d_input)
        cuda_lib.cuda_free(d_input)
        
        # Flatten
        out = out.reshape(batch_size, -1)
        
        # FC layers
        d_input = cuda_lib.cuda_malloc(out.size)
        cuda_lib.cuda_memcpy_h2d(
            d_input,
            out.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.size
        )
        cuda_lib.cuda_free(d_out)
        
        out, d_out = self.fc1.forward(out, d_input)
        cuda_lib.cuda_free(d_input)
        
        out = np.maximum(0, out)
        
        d_input = cuda_lib.cuda_malloc(out.size)
        cuda_lib.cuda_memcpy_h2d(
            d_input,
            out.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.size
        )
        cuda_lib.cuda_free(d_out)
        
        out, d_out = self.fc2.forward(out, d_input)
        cuda_lib.cuda_free(d_input)
        cuda_lib.cuda_free(d_out)
        
        # Sigmoid
        out = 1 / (1 + np.exp(-np.clip(out, -500, 500)))
        
        return out
    
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
    """Training loop for CUDA CNN"""
    
    def __init__(self, model):
        self.model = model
    
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
            
            predictions = self.model.forward(X_batch)
            
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
        print("CUDA-ACCELERATED CNN TRAINING")
        print("="*60)
        
        total_start = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            train_loss, train_acc = self.train_epoch(X_train, y_train, batch_size)
            val_loss, val_acc = self.evaluate(X_val, y_val, batch_size)
            
            epoch_time = time.time() - epoch_start
            
            print(f"\nEpoch {epoch+1}/{epochs} - Time: {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        total_time = time.time() - total_start
        
        print("\n" + "="*60)
        print(f"✓ CUDA training completed in {total_time:.2f} seconds")
        print(f"✓ Average time per epoch: {total_time/epochs:.2f} seconds")
        print("="*60)
        
        return total_time


def main():
    """Main function"""
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "processed"
    RESULTS_DIR = BASE_DIR / "results" / "gpu_benchmarks"
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
    
    # Create CUDA model
    model = SimpleCNN_CUDA()
    trainer = Trainer(model)
    
    # Train
    training_time = trainer.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=16)
    
    # Save results
    with open(RESULTS_DIR / "cuda_timing.txt", "w") as f:
        f.write(f"CUDA Training Time: {training_time:.2f} seconds\n")
        f.write(f"Average per epoch: {training_time/5:.2f} seconds\n")
    
    print("\n✓ CUDA TRAINING COMPLETED!")


if __name__ == "__main__":
    main()