# Brain Tumor Classification System  
**A Deep Learning-Based Web Application for MRI Analysis**  

---

## 📌 Overview  
This project is a **Flask-based web application** that uses a **deep learning model (ResNet18)** to classify brain tumors from MRI scans into four categories:  
- **Glioma Tumor**  
- **Meningioma Tumor**  
- **Pituitary Tumor**  
- **No Tumor**  

The system provides:  
✅ **Automated tumor detection**  
✅ **Medical insights** (causes, symptoms, treatment options)  
✅ **Image preprocessing visualization** (grayscale, noise reduction, contrast enhancement)  
✅ **Confidence scoring** for predictions  

---

## ⚙️ Technical Stack  
| Component       | Technology               |  
|-----------------|--------------------------|  
| **Backend**     | Python (Flask)           |  
| **Deep Learning** | PyTorch (ResNet18)       |  
| **Image Processing** | OpenCV, scikit-image |  
| **Frontend**    | HTML/CSS/JavaScript      |  

---

## 🚀 Features  
### 1. Image Preprocessing Pipeline  
- **Grayscale Conversion** → **Noise Reduction** → **Contrast Enhancement** → **Thresholding**  
- Visualizes each step for transparency.  

### 2. AI-Powered Classification  
- Returns **diagnosis + confidence score** (0-100%).  

### 3. Medical Knowledge Integration  
- Provides **treatment options, prognosis, and follow-up guidelines**.  

### 4. Secure File Handling  
- Uses `secure_filename` to prevent malicious uploads.  

