import os

folder = r"C:\Users\shree\Documents\cuda-medical-sim\brain-tumor-classification-mri"
extensions = {
    "Python files": ['.py'],
    "CSV files": ['.csv'],
    "PTH files": ['.pth'],
    "Image files": ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
}

counts = {key: 0 for key in extensions}

for root, dirs, files in os.walk(folder):
    for file in files:
        for key, exts in extensions.items():
            if any(file.lower().endswith(ext) for ext in exts):
                counts[key] += 1

for key, count in counts.items():
    print(f"{key}: {count}")

import os

# Root folder path
root_folder = r"C:\Users\shree\Documents\cuda-medical-sim\brain-tumor-classification-mri"
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

# Walk through all folders
for dirpath, dirnames, filenames in os.walk(root_folder):
    image_count = sum(1 for file in filenames if file.lower().endswith(image_extensions))
    if image_count > 0:
        print(f"{dirpath} - {image_count} image(s)")
import os

# Root directory to search
root_dir = r"C:\Users\shree\Documents\cuda-medical-sim\brain-tumor-classification-mri"
target_extensions = ('.py', '.csv', '.pth')

# Dictionary to store files by extension
files_by_type = {"py": [], "csv": [], "pth": []}

# Walk through all subdirectories
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith(target_extensions):
            ext = file.split(".")[-1]
            files_by_type[ext].append(os.path.join(dirpath, file))

# Print the results
for ext, files in files_by_type.items():
    print(f"\n--- .{ext} Files ---")
    for file_path in files:
        print(file_path)
