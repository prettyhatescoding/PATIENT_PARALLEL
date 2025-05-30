import os
import time
import torch
import torch.nn as nn
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from torchvision import models, transforms
from PIL import Image
from skimage import exposure
from skimage.feature import graycomatrix, graycoprops

app = Flask(__name__)


UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


TUMOR_INFO = {
    'glioma': {
        'description': 'Gliomas are tumors that arise from glial cells in the brain and spinal cord.',
        'causes': 'Genetic factors, radiation exposure, certain genetic disorders',
        'symptoms': 'Headaches, nausea, vomiting, seizures, memory loss, personality changes',
        'first_line': 'Surgical resection followed by radiation and chemotherapy (temozolomide)',
        'alternatives': 'Targeted therapy, immunotherapy, clinical trials',
        'non_surgical': 'Radiation therapy, chemotherapy, tumor treating fields (TTF)',
        'follow_up': 'MRI every 2-3 months initially, then every 6 months if stable',
        'prognosis': 'Varies by grade (II-IV), with 5-year survival ranging from 10-90%'
    },
    'meningioma': {
        'description': 'Meningiomas are tumors arising from the meninges, the membranes surrounding the brain.',
        'causes': 'Radiation exposure, female hormones, neurofibromatosis type 2',
        'symptoms': 'Headaches, seizures, vision problems, arm/leg weakness',
        'first_line': 'Surgical resection, often curative for grade I tumors',
        'alternatives': 'Radiosurgery for small tumors, hormone therapy',
        'non_surgical': 'Observation for asymptomatic cases, radiation therapy',
        'follow_up': 'Annual MRI for 5 years, then every 2-3 years',
        'prognosis': 'Excellent for grade I (90% 10-year survival), poorer for higher grades'
    },
    'pituitary': {
        'description': 'Pituitary adenomas are tumors of the pituitary gland at the base of the brain.',
        'causes': 'Most are sporadic, some genetic syndromes (MEN1, Carney complex)',
        'symptoms': 'Vision problems, headaches, hormonal imbalances',
        'first_line': 'Transsphenoidal surgical resection',
        'alternatives': 'Medical therapy (dopamine agonists for prolactinomas)',
        'non_surgical': 'Medication to control hormone secretion, radiation therapy',
        'follow_up': 'Hormonal evaluation and MRI at 3-6 months post-op',
        'prognosis': 'Good for microadenomas (>90% cure), variable for macroadenomas'
    },
    'no_tumor': {
        'description': 'No evidence of tumor detected in the MRI scan.',
        'causes': 'N/A',
        'symptoms': 'N/A',
        'recommendations': 'Continue routine care, follow up if symptoms develop',
        'follow_up': 'None required unless clinical suspicion arises',
        'prognosis': 'Excellent'
    }
}

class BrainTumorClassifier:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path):
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 4)
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def process_image(self, image_path):
        try:
            
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image")
            
            base_filename = secure_filename(os.path.basename(image_path))
            base_name = os.path.splitext(base_filename)[0]
            results = {'original': self._save_image(img, f"{base_name}_original.jpg")}
            
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            results['gray'] = self._save_image(gray, f"{base_name}_gray.jpg")
            
            
            blurred = cv2.GaussianBlur(gray, (5,5), 0)
            results['blurred'] = self._save_image(blurred, f"{base_name}_blurred.jpg")
            
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(blurred)
            results['enhanced'] = self._save_image(enhanced, f"{base_name}_enhanced.jpg")
            
            
            thresholded = cv2.adaptiveThreshold(enhanced, 255, 
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
            results['thresholded'] = self._save_image(thresholded, f"{base_name}_thresholded.jpg")
            
            
            classification = self._classify_image(image_path)
            
            return {
                'success': True,
                'processing_steps': results,
                'classification': classification,
                'medical_info': TUMOR_INFO[classification['diagnosis']]
            }
            
        except Exception as e:
            print(f"Processing error: {e}")
            return {'success': False, 'error': str(e)}

    def _classify_image(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
            
            return {
                'diagnosis': self.class_names[pred.item()],
                'confidence': round(conf.item() * 100, 2)
            }
        except Exception as e:
            print(f"Classification error: {e}")
            return {'diagnosis': 'error', 'confidence': 0}

    def _save_image(self, image, filename):
        """Save processing step image"""
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if len(image.shape) == 2:  
            cv2.imwrite(path, image)
        else:  
            cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return filename


classifier = BrainTumorClassifier('models/best_model.pth')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        
       
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        
        result = classifier.process_image(filepath)
        
        if not result['success']:
            return jsonify({'error': result.get('error', 'Processing failed')}), 500
            
        return jsonify({
            'diagnosis': result['classification']['diagnosis'],
            'confidence': result['classification']['confidence'],
            'processing_time': f"{(time.time()-start_time)*1000:.0f}ms",
            'processing_steps': result['processing_steps'],
            'medical_info': result['medical_info']
        })
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)