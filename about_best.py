import torch

# Load the checkpoint
checkpoint_path = r"C:\Users\shree\Documents\cuda-medical-sim\brain-tumor-classification-mri\models\best_model.pth"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Force CPU if no GPU

# Print keys to see what's saved
print("Keys in the checkpoint:", checkpoint.keys())

# If 'accuracy' or 'val_accuracy' is stored
if 'accuracy' in checkpoint:
    print(f"Best Accuracy: {checkpoint['accuracy']:.4f}")

# If 'epoch' is stored
if 'epoch' in checkpoint:
    print(f"Last Epoch: {checkpoint['epoch']}")

import matplotlib.pyplot as plt

# Example: Plot accuracy/loss curves (if saved in checkpoint)
if 'history' in checkpoint:
    history = checkpoint['history']
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()