import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from PIL import Image

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data', 'processed8')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'wing_classifier_convnext.pth')
SAVE_PATH = os.path.join(PROJECT_ROOT, 'confusion_matrix_convnext.png')

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET ---
class FlyWingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        try:
            self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        except FileNotFoundError:
            print(f"❌ Error: Data directory not found at {root_dir}")
            self.classes = []
            
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
        if img_cv is None: raise ValueError(f"Failed: {path}")
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        if self.transform: img_pil = self.transform(img_pil)
        return img_pil, label

def plot_confusion_matrix(cm, classes, title='Confusion Matrix - ConvNeXt'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"\n✅ Confusion Matrix saved to: {SAVE_PATH}")
    plt.close()

def evaluate():
    print(f"🔍 Evaluating ConvNeXt on {DEVICE}...")
    
    # 1. SETUP
    # Note: No robust transforms for eval, just resize/normalize
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = FlyWingDataset(DATA_DIR, transform=eval_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 2. LOAD MODEL
    print("⚙️ Loading Model...")
    model = models.convnext_tiny(weights=None)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, len(dataset.classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # 3. PREDICT
    y_true = []
    y_pred = []
    
    print("🚀 Running Inference...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    # 4. REPORT
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, dataset.classes)
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=dataset.classes, digits=4))

if __name__ == "__main__":
    evaluate()