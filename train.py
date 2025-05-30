import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import time
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "Training")
TEST_DIR = os.path.join(BASE_DIR, "Testing")
BATCH_SIZE = 32
IMAGE_SIZE = 224
MAX_EPOCHS = 20
LEARNING_RATE = 0.001
PATIENCE = 3
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")
HISTORY_PATH = os.path.join(os.path.expanduser("~"), "brain_tumor_training_history.csv")  # Changed to user directory
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = []
        self.images = []
        self.labels = []
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory not found: {data_dir}")
        
        self.classes = sorted([d for d in os.listdir(data_dir) 
                             if os.path.isdir(os.path.join(data_dir, d))])
        
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            images = [os.path.join(class_dir, img) 
                     for img in os.listdir(class_dir) 
                     if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.images.extend(images)
            self.labels.extend([idx] * len(images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), 0

def get_data_loaders():
    """Create and return train and test data loaders"""
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = BrainTumorDataset(TRAIN_DIR, transform=train_transform)
    test_dataset = BrainTumorDataset(TEST_DIR, transform=test_transform)
    
    print("\nDataset Summary:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}\n")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, test_loader, train_dataset.classes


class TrainingManager:
    def __init__(self, model_path):
        self.model_path = model_path
        self.best_acc = 0.0
        self.best_epoch = 0
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 4)
        
        if os.path.exists(model_path):
            print("Loading existing model...")
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self._load_history()
        else:
            print("Initializing new model...")
        
        self.model = self.model.to(device)
    
    def _load_history(self):
        """Load training history with robust error handling"""
        try:
            if os.path.exists(HISTORY_PATH):
                with open(HISTORY_PATH, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        all_accs = [float(line.split(',')[-1]) for line in lines[1:]]
                        self.best_acc = max(all_accs)
                        self.best_epoch = all_accs.index(self.best_acc) + 1
                        print(f"Previous best: {self.best_acc:.2f}% (epoch {self.best_epoch})")
        except Exception as e:
            print(f"Warning: Could not load history - {e}")
            self.best_acc = 0.0
            self.best_epoch = 0
    
    def save_model(self, current_acc, current_epoch):
        if current_acc > self.best_acc:
            self.best_acc = current_acc
            self.best_epoch = current_epoch
            try:
                torch.save(self.model.state_dict(), self.model_path)
                print(f" New best model! Accuracy: {current_acc:.2f}% (epoch {current_epoch})")
                return True
            except Exception as e:
                print(f"Error saving model: {e}")
        return False
    
    def log_training(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Log training metrics with permission handling"""
        log_entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{epoch},{train_loss:.4f},{train_acc:.2f},{val_loss:.4f},{val_acc:.2f}\n"
        
        try:
            with open(HISTORY_PATH, 'a' if os.path.exists(HISTORY_PATH) else 'w') as f:
                if not os.path.exists(HISTORY_PATH):
                    f.write("timestamp,epoch,train_loss,train_acc,val_loss,val_acc\n")
                f.write(log_entry)
        except PermissionError:
            alt_path = os.path.join(BASE_DIR, "temp_training_history.csv")
            print(f"Warning: Could not write to {HISTORY_PATH}, using {alt_path} instead")
            with open(alt_path, 'a') as f:
                if not os.path.exists(alt_path):
                    f.write("timestamp,epoch,train_loss,train_acc,val_loss,val_acc\n")
                f.write(log_entry)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(loader), 100 * correct / total

def main():
    train_loader, test_loader, classes = get_data_loaders()
    manager = TrainingManager(MODEL_PATH)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(manager.model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nStarting training (Best: {manager.best_acc:.2f}%)")
    print("="*50)
    
    patience_counter = 0
    for epoch in range(MAX_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_epoch(manager.model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(manager.model, test_loader, criterion)
        epoch_time = time.time() - start_time
        
        manager.log_training(epoch+1, train_loss, train_acc, val_loss, val_acc)
        
        print(f"\nEpoch {epoch+1}/{MAX_EPOCHS} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        if manager.save_model(val_acc, epoch+1):
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("\nEarly stopping!")
                break
    
    print("\nTraining complete!")
    print(f"Best accuracy: {manager.best_acc:.2f}% (epoch {manager.best_epoch})")

if __name__ == "__main__":
    main()