import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import time
from tkinter import Tk, filedialog

class BrainTumorDetector:
    def __init__(self, model_path):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("\n" + "="*50)
        print(f"{'Brain Tumor Detector Initialization':^50}")
        print("="*50)
        print(f"Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA: {torch.version.cuda}")
        else:
            print("Running on CPU (CUDA not available)")
        print("="*50 + "\n")

       
        self.model = self._load_model(model_path)
        self.class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path):
        """Load model with progress logging"""
        print("Loading model...")
        start_time = time.time()
        
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 4)
        
        if os.path.exists(model_path):
            print(f"Loading weights from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded in {time.time()-start_time:.2f}s")
        print(f"Model on: {next(model.parameters()).device}\n")
        return model

    def analyze_image(self, image_path):
        """Analyze single image with detailed logging"""
        print(f"\nAnalyzing: {os.path.basename(image_path)}")
        start_time = time.time()
        
        try:
            
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
            
            
            diagnosis = self.class_names[pred.item()]
            confidence = round(conf.item() * 100, 2)
            
            
            print("\n" + "="*50)
            print(f"{' ANALYSIS RESULTS ':-^50}")
            print(f"Diagnosis: {diagnosis.replace('_', ' ').upper()}")
            print(f"Confidence: {confidence}%")
            print(f"Processing time: {time.time()-start_time:.2f}s")
            if self.device.type == 'cuda':
                print(f"GPU Memory used: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            print("="*50)
            
            return diagnosis, confidence
            
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            return None, None

    def analyze_folder(self, folder_path):
        """Analyze all valid images in a folder"""
        print(f"\nScanning folder: {folder_path}")
        
        
        valid_ext = ('.png', '.jpg', '.jpeg')
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(valid_ext)]
        
        if not image_files:
            print("No valid images found (supported: .png, .jpg, .jpeg)")
            return
        
        print(f"Found {len(image_files)} images to process\n")
        
       
        results = []
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            diagnosis, confidence = self.analyze_image(img_path)
            if diagnosis:
                results.append((img_file, diagnosis, confidence))
        
        
        if results:
            print("\n" + "="*50)
            print(f"{' PROCESSING SUMMARY ':-^50}")
            for img_file, diagnosis, confidence in results:
                print(f"{img_file:<30} {diagnosis.upper():<15} {confidence}%")
            print("="*50)

def select_file_or_folder():
    """GUI dialog to select file or folder"""
    root = Tk()
    root.withdraw() 
    
    print("\nSelect analysis mode:")
    print("1. Single image file")
    print("2. Folder with multiple images")
    choice = input("Enter choice (1/2): ").strip()
    
    if choice == '1':
        file_path = filedialog.askopenfilename(
            title="Select MRI Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        return file_path if file_path else None
    elif choice == '2':
        folder_path = filedialog.askdirectory(title="Select Folder with MRI Images")
        return folder_path if folder_path else None
    return None

if __name__ == '__main__':
    
    detector = BrainTumorDetector('models/best_model.pth')
    
    while True:
       
        path = select_file_or_folder()
        if not path:
            print("No selection made. Try again.")
            continue
        
        
        if os.path.isfile(path):
            detector.analyze_image(path)
        else:
            detector.analyze_folder(path)
        
       
        if input("\nAnalyze more? (y/n): ").lower() != 'y':
            break
    
    print("\nAnalysis complete. Exiting...")