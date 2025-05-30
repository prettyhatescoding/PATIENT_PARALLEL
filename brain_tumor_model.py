import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import cv2
import numpy as np

class BrainTumorClassifier:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        self.transform = self._get_transforms()

    def _load_model(self, model_path):
        """Load the trained ResNet18 model"""
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 4)
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model = model.to(self.device)
        model.eval()
        return model
        
    def _get_transforms(self):
        """Image preprocessing transformations"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def predict(self, image_path):
        """Make prediction on an image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                _, preds = torch.max(outputs, 1)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence = torch.max(probs).item()
            
            prediction = self.classes[preds.item()]
            
            
            if prediction != 'no_tumor':
                contour_filename = self._generate_contours(image_path)
            else:
                contour_filename = None
                
            return prediction, round(confidence * 100, 2), contour_filename
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "error", 0.0, None
            
    def _generate_contours(self, image_path):
        """Generate MRI scan with tumor contours"""
        try:
           
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
           
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 10000:  
                    filtered_contours.append(contour)
            
           
            contour_img = img.copy()
            cv2.drawContours(contour_img, filtered_contours, -1, (0,255,0), 2)
            
            
            filename = os.path.basename(image_path)
            base_name, ext = os.path.splitext(filename)
            contour_filename = f"{base_name}_contour{ext}"
            contour_path = os.path.join(os.path.dirname(image_path), contour_filename)
            cv2.imwrite(contour_path, contour_img)
            
            return contour_filename
            
        except Exception as e:
            print(f"Contour generation error: {e}")
            return None