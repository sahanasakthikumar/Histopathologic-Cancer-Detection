"""
Optimized CNN with Backpropagation - Actually Learns!
This version includes gradient descent to update weights
"""

import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt

class ActivationFunctions:
    """Activation functions and their derivatives"""
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)


class ConvLayer:
    """Optimized Convolutional Layer with Backpropagation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.bias = np.zeros(out_channels)
        
        self.weights_col = self.weights.reshape(out_channels, -1)
        
        # For backprop
        self.input = None
        self.col = None
        self.output = None
        self.dweights = None
        self.dbias = None
    
    def im2col(self, x, out_h, out_w):
        """Convert input to column matrix"""
        batch_size, channels, h, w = x.shape
        
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), 
                          (self.padding, self.padding)), mode='constant')
        
        col = np.zeros((batch_size, channels, self.kernel_size, self.kernel_size, out_h, out_w))
        
        for y in range(self.kernel_size):
            y_max = y + self.stride * out_h
            for x_idx in range(self.kernel_size):
                x_max = x_idx + self.stride * out_w
                col[:, :, y, x_idx, :, :] = x[:, :, y:y_max:self.stride, x_idx:x_max:self.stride]
        
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * out_h * out_w, -1)
        return col
    
    def forward(self, x):
        """Forward pass"""
        self.input = x
        batch_size, _, h, w = x.shape
        
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        self.col = self.im2col(x, out_h, out_w)
        out = np.dot(self.col, self.weights_col.T) + self.bias
        out = out.reshape(batch_size, out_h, out_w, -1).transpose(0, 3, 1, 2)
        
        self.output = out
        return out
    
    def backward(self, dout, learning_rate):
        """Backward pass - updates weights"""
        batch_size, out_channels, out_h, out_w = dout.shape
        
        # Reshape gradient
        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        
        # Compute gradients
        self.dweights = np.dot(dout_reshaped.T, self.col).reshape(self.weights.shape)
        self.dbias = np.sum(dout_reshaped, axis=0)
        
        # Update weights (gradient descent)
        self.weights -= learning_rate * self.dweights / batch_size
        self.bias -= learning_rate * self.dbias / batch_size
        
        # Update reshaped weights
        self.weights_col = self.weights.reshape(self.out_channels, -1)
        
        # Compute gradient for previous layer
        dx_col = np.dot(dout_reshaped, self.weights_col)
        
        # Reshape back to image format
        dx = self.col2im(dx_col, self.input.shape, out_h, out_w)
        return dx
    
    def col2im(self, col, input_shape, out_h, out_w):
        """Convert column matrix back to image format"""
        batch_size, channels, h, w = input_shape
        
        if self.padding > 0:
            h_padded = h + 2 * self.padding
            w_padded = w + 2 * self.padding
        else:
            h_padded = h
            w_padded = w
        
        dx = np.zeros((batch_size, channels, h_padded, w_padded))
        col = col.reshape(batch_size, out_h, out_w, channels, self.kernel_size, self.kernel_size)
        col = col.transpose(0, 3, 4, 5, 1, 2)
        
        for y in range(self.kernel_size):
            y_max = y + self.stride * out_h
            for x_idx in range(self.kernel_size):
                x_max = x_idx + self.stride * out_w
                dx[:, :, y:y_max:self.stride, x_idx:x_max:self.stride] += col[:, :, y, x_idx, :, :]
        
        # Remove padding if it was added
        if self.padding > 0:
            dx = dx[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        return dx


class MaxPoolLayer:
    """Max Pooling with Backpropagation"""
    
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.output = None
        self.max_indices = None
    
    def forward(self, x):
        """Forward pass"""
        self.input = x
        batch_size, channels, h, w = x.shape
        
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        
        x_reshaped = x.reshape(batch_size, channels, out_h, self.pool_size, 
                               out_w, self.pool_size)
        output = x_reshaped.max(axis=3).max(axis=4)
        
        self.output = output
        return output
    
    def backward(self, dout):
        """Backward pass - distribute gradient to max elements"""
        batch_size, channels, out_h, out_w = dout.shape
        _, _, h, w = self.input.shape
        
        dx = np.zeros_like(self.input)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size
                        
                        window = self.input[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(window)
                        mask = (window == max_val)
                        dx[b, c, h_start:h_end, w_start:w_end] += dout[b, c, i, j] * mask
        
        return dx


class FullyConnectedLayer:
    """FC Layer with Backpropagation"""
    
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros(output_size)
        
        self.input = None
        self.output = None
    
    def forward(self, x):
        """Forward pass"""
        self.input = x
        self.output = np.dot(x, self.weights) + self.bias
        return self.output
    
    def backward(self, dout, learning_rate):
        """Backward pass - updates weights"""
        batch_size = self.input.shape[0]
        
        # Compute gradients
        dweights = np.dot(self.input.T, dout)
        dbias = np.sum(dout, axis=0)
        
        # Update weights
        self.weights -= learning_rate * dweights / batch_size
        self.bias -= learning_rate * dbias / batch_size
        
        # Gradient for previous layer
        dx = np.dot(dout, self.weights.T)
        return dx


class SimpleCNN:
    """CNN with Full Backpropagation"""
    
    def __init__(self, input_shape=(3, 96, 96)):
        self.input_shape = input_shape
        
        self.conv1 = ConvLayer(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = MaxPoolLayer(pool_size=2, stride=2)
        
        self.conv2 = ConvLayer(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = MaxPoolLayer(pool_size=2, stride=2)
        
        self.flat_size = 32 * 24 * 24
        
        self.fc1 = FullyConnectedLayer(self.flat_size, 128)
        self.fc2 = FullyConnectedLayer(128, 1)
        
        self.activation = ActivationFunctions()
        
        # Cache activations for backprop
        self.cache = {}
    
    def forward(self, x):
        """Forward pass - saves intermediate values"""
        # Conv1 -> ReLU -> Pool1
        out = self.conv1.forward(x)
        self.cache['conv1_out'] = out
        out = self.activation.relu(out)
        self.cache['relu1_out'] = out
        out = self.pool1.forward(out)
        
        # Conv2 -> ReLU -> Pool2
        out = self.conv2.forward(out)
        self.cache['conv2_out'] = out
        out = self.activation.relu(out)
        self.cache['relu2_out'] = out
        out = self.pool2.forward(out)
        
        # Flatten
        batch_size = out.shape[0]
        out = out.reshape(batch_size, -1)
        self.cache['pool2_shape'] = self.pool2.output.shape
        
        # FC1 -> ReLU
        out = self.fc1.forward(out)
        self.cache['fc1_out'] = out
        out = self.activation.relu(out)
        
        # FC2 -> Sigmoid
        out = self.fc2.forward(out)
        self.cache['fc2_out'] = out
        out = self.activation.sigmoid(out)
        
        return out
    
    def backward(self, predictions, targets, learning_rate):
        """Backward pass - the KEY to learning!"""
        batch_size = predictions.shape[0]
        
        # Output gradient (derivative of BCE loss w.r.t. predictions)
        dout = predictions - targets
        
        # Backprop through FC2
        dout = dout * self.activation.sigmoid_derivative(self.cache['fc2_out'])
        dout = self.fc2.backward(dout, learning_rate)
        
        # Backprop through ReLU + FC1
        dout = dout * self.activation.relu_derivative(self.cache['fc1_out'])
        dout = self.fc1.backward(dout, learning_rate)
        
        # Reshape for conv layers
        dout = dout.reshape(self.cache['pool2_shape'])
        
        # Backprop through Pool2
        dout = self.pool2.backward(dout)
        
        # Backprop through ReLU + Conv2
        dout = dout * self.activation.relu_derivative(self.cache['conv2_out'])
        dout = self.conv2.backward(dout, learning_rate)
        
        # Backprop through Pool1
        dout = self.pool1.backward(dout)
        
        # Backprop through ReLU + Conv1
        dout = dout * self.activation.relu_derivative(self.cache['conv1_out'])
        self.conv1.backward(dout, learning_rate)
    
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
    """Training loop with backpropagation"""
    
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
    
    def train_epoch(self, X_train, y_train, batch_size=32):
        """Train for one epoch WITH weight updates"""
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
            
            # Compute loss and accuracy
            loss = self.model.compute_loss(predictions, y_batch)
            acc = self.model.compute_accuracy(predictions, y_batch)
            
            # BACKWARD PASS - This is what was missing!
            self.model.backward(predictions, y_batch, self.learning_rate)
            
            epoch_loss += loss
            epoch_acc += acc
            n_batches += 1
            
            if (n_batches % 10 == 0):
                print(f"  Batch {n_batches}/{(n_samples + batch_size - 1) // batch_size} - Loss: {loss:.4f}, Acc: {acc:.2f}%", end='\r')
        
        print()
        return epoch_loss / n_batches, epoch_acc / n_batches
    
    def evaluate(self, X_val, y_val, batch_size=32):
        """Evaluate without updating weights"""
        predictions = self.model.forward(X_val[:batch_size])
        y_batch = y_val[:batch_size].reshape(-1, 1)
        
        loss = self.model.compute_loss(predictions, y_batch)
        acc = self.model.compute_accuracy(predictions, y_batch)
        
        return loss, acc
    
    def train(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=32):
        """Training loop"""
        print("\n" + "="*60)
        print("CNN TRAINING WITH BACKPROPAGATION")
        print("="*60)
        print(f"Model: SimpleCNN (with gradient descent)")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print("="*60)
        
        total_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(X_train, y_train, batch_size)
            val_loss, val_acc = self.evaluate(X_val, y_val, batch_size)
            
            epoch_time = time.time() - epoch_start_time
            
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            print(f"\nEpoch {epoch+1}/{epochs} - Time: {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        total_time = time.time() - total_start_time
        
        print("\n" + "="*60)
        print(f"✓ Training completed in {total_time:.2f} seconds")
        print(f"✓ Average time per epoch: {total_time/epochs:.2f} seconds")
        print("="*60)
        
        return total_time
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.train_losses, label='Train Loss', marker='o')
        ax1.plot(self.val_losses, label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.train_accs, label='Train Acc', marker='o')
        ax2.plot(self.val_accs, label='Val Acc', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Training plot saved to: {save_path}")
        
        plt.show()


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
    
    print(f"✓ Train set: {X_train.shape}")
    print(f"✓ Val set: {X_val.shape}")
    
    # Use subset
    TRAIN_SUBSET = 1000
    VAL_SUBSET = 200
    
    X_train = X_train[:TRAIN_SUBSET]
    y_train = y_train[:TRAIN_SUBSET]
    X_val = X_val[:VAL_SUBSET]
    y_val = y_val[:VAL_SUBSET]
    
    print(f"\nUsing: {len(X_train)} train, {len(X_val)} val samples")
    
    model = SimpleCNN(input_shape=(3, 96, 96))
    trainer = Trainer(model, learning_rate=0.01)  # Higher LR for faster learning
    
    training_time = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=5,
        batch_size=16
    )
    
    plot_path = RESULTS_DIR / "training_with_backprop.png"
    trainer.plot_training_history(save_path=plot_path)
    
    with open(RESULTS_DIR / "training_results.txt", "w") as f:
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Final Train Acc: {trainer.train_accs[-1]:.2f}%\n")
        f.write(f"Final Val Acc: {trainer.val_accs[-1]:.2f}%\n")
    
    print("\n✓ TRAINING WITH BACKPROPAGATION COMPLETED!")


if __name__ == "__main__":
    main()