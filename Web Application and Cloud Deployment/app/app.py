"""
Flask Web Application for Histopathology Cancer Detection
Loads both CPU and GPU models and compares predictions
"""

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import io
import time
from pathlib import Path
import json

app = Flask(__name__)

# Model architecture (same as training)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# Load models
MODELS_DIR = Path(__file__).parent.parent / "models"

def load_models():
    """Load both CPU and GPU models"""
    models = {}
    
    # Load CPU model
    cpu_model = SimpleCNN()
    cpu_checkpoint = torch.load(MODELS_DIR / "cpu_model.pth", map_location='cpu')
    cpu_model.load_state_dict(cpu_checkpoint['model_state_dict'])
    cpu_model.eval()
    models['cpu'] = cpu_model
    
    # Load GPU model
    if torch.cuda.is_available():
        gpu_model = SimpleCNN()
        gpu_checkpoint = torch.load(MODELS_DIR / "gpu_model.pth", map_location='cuda')
        gpu_model.load_state_dict(gpu_checkpoint['model_state_dict'])
        gpu_model.to('cuda')
        gpu_model.eval()
        models['gpu'] = gpu_model
    
    # Load metadata
    with open(MODELS_DIR / "training_metadata.json", 'r') as f:
        models['metadata'] = json.load(f)
    
    return models

print("Loading models...")
MODELS = load_models()
print("✓ Models loaded successfully")

def preprocess_image(image_file):
    """Preprocess uploaded image"""
    img = Image.open(io.BytesIO(image_file.read()))
    
    # Convert to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to 96x96
    img = img.resize((96, 96), Image.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Transpose to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension
    img_tensor = torch.FloatTensor(img_array).unsqueeze(0)
    
    return img_tensor

def predict_with_model(model, image_tensor, device='cpu'):
    """Make prediction with given model"""
    start_time = time.time()
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probability = output.item()
        prediction = 1 if probability >= 0.5 else 0
    
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    return {
        'prediction': int(prediction),
        'probability': float(probability),
        'confidence': float(probability if prediction == 1 else 1 - probability),
        'inference_time_ms': round(inference_time, 2),
        'label': 'Cancer Detected' if prediction == 1 else 'No Cancer'
    }

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', 
                         metadata=MODELS['metadata'],
                         gpu_available=torch.cuda.is_available())

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Preprocess image
        image_tensor = preprocess_image(file)
        
        # Predict with CPU model
        cpu_result = predict_with_model(MODELS['cpu'], image_tensor, device='cpu')
        
        # Predict with GPU model if available
        if 'gpu' in MODELS:
            gpu_result = predict_with_model(MODELS['gpu'], image_tensor, device='cuda')
        else:
            gpu_result = None
        
        # Compare results
        comparison = {
            'cpu': cpu_result,
            'gpu': gpu_result,
            'speedup': None
        }
        
        if gpu_result:
            comparison['speedup'] = round(cpu_result['inference_time_ms'] / gpu_result['inference_time_ms'], 2)
            comparison['agreement'] = cpu_result['prediction'] == gpu_result['prediction']
        
        return jsonify(comparison)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def stats():
    """Get model statistics"""
    return jsonify(MODELS['metadata'])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)