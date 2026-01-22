import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data', 'processed8')

# POINTING TO THE RESNET GPU MODEL
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'wing_classifier_resnet_gpu.pth')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CUSTOM DATASET (Same as training) ---
class FlyWingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.tif', '.png', '.jpg')):
                    self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img_cv = cv2.imread(path)
        if img_cv is None:
            raise ValueError(f"Failed to load image: {path}")
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        if self.transform:
            img_pil = self.transform(img_pil)
        return img_pil, label

def generate_report():
    print(f"📊 Loading ResNet50 from {MODEL_PATH}...")
    
    # 1. SETUP DATA (Critical: 224x224 for ResNet)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = FlyWingDataset(DATA_DIR, transform=val_transforms)
    # num_workers=0 to prevent Windows crash
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"   Analyzing {len(dataset)} images...")

    # 2. LOAD ARCHITECTURE (ResNet50)
    model = models.resnet50(weights=None)
    # ResNet uses 'fc' layer, NOT 'classifier'
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(dataset.classes))
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("   ✅ Weights loaded successfully.")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return

    # 3. RUN INFERENCE
    all_preds = []
    all_labels = []
    
    print("🚀 Running Inference...")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. REPORT & MATRIX
    print("\n" + "="*60)
    print(f"      RESNET50 REPORT (Accuracy: {np.mean(np.array(all_preds) == np.array(all_labels)):.2%})")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=dataset.classes))

    # 5. PLOT CONFUSION MATRIX
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    # Using 'Blues' to distinguish from EfficientNet's 'Greens'
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.xlabel('Predicted Species')
    plt.ylabel('Actual Species')
    plt.title('Confusion Matrix - ResNet50')
    plt.tight_layout()
    
    save_path = os.path.join(PROJECT_ROOT, 'confusion_matrix_resnet.png')
    plt.savefig(save_path)
    print(f"\n✅ Matrix saved to: {save_path}")

if __name__ == "__main__":
    generate_report()