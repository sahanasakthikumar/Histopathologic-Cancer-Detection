---

## **3. GPU Acceleration**
```markdown
# GPU Acceleration with CUDA

## Overview
Implements CNN operations using NVIDIA CUDA for GPU parallelization. Achieves significant speedup over CPU implementations.

## Files
- `conv_cuda.cu` - CUDA kernels for convolution, pooling, and matrix operations
- `cnn_cuda.py` - Python interface to CUDA functions
- `train_and_save_cuda.py` - GPU-accelerated training

## Key Features
- CUDA kernels for im2col transformation
- GPU matrix multiplication
- Parallel activation functions (ReLU, Sigmoid)
- GPU memory management

## Requirements
- NVIDIA GPU with CUDA support (tested on RTX 3050)
- CUDA Toolkit 12.7+

