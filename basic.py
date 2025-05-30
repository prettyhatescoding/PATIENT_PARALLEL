import os
import time
import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from functools import lru_cache

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Correct model architecture (ResNet-based)
def create_model():
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 4)  # 4 output classes
    return model

# Model loader
@lru_cache(maxsize=1)
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model()
    
    try:
        state_dict = torch.load('models/best_model.pth', map_location=device)
        
        # Handle DataParallel and other potential key mismatches
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # Remove 'module.' prefix if present
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        model.to(device).eval()
        print(f"Model loaded successfully on {device}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

# Initialize model
try:
    model = load_model()
except Exception as e:
    print(f"Using randomly initialized model due to error: {str(e)}")
    model = create_model().eval()

class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_stream):
    # Read and decode image
    img = cv2.imdecode(
        np.frombuffer(file_stream.read(), np.uint8),
        cv2.IMREAD_COLOR
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Normalize
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = (img_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                 torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img_tensor.unsqueeze(0)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
        
    try:
        start_time = time.time()
        img_tensor = preprocess_image(file)
        
        with torch.no_grad():
            outputs = model(img_tensor.to(next(model.parameters()).device))
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        
        # Generate simple heatmap
        heatmap = torch.mean(img_tensor.squeeze(), dim=0).numpy()
        heatmap = (heatmap * 255).astype(np.uint8)
        
        return jsonify({
            'diagnosis': class_names[pred.item()],
            'confidence': f"{conf.item()*100:.1f}%",
            'processing_time': f"{(time.time()-start_time)*1000:.0f}ms",
            'heatmap': heatmap.tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)