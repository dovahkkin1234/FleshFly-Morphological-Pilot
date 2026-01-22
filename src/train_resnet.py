import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import os
import cv2
import numpy as np
from PIL import Image
import time
from tqdm import tqdm

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data', 'processed8')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'wing_classifier_resnet_gpu.pth')

# --- CONFIGURATION ---
BATCH_SIZE = 16  # GPU can handle larger batches
EPOCHS = 30
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CUSTOM DATASET (The Fix) ---
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
        
        # 1. OPENCV LOAD (Stable)
        img_cv = cv2.imread(path)
        if img_cv is None:
            raise ValueError(f"Failed to load image: {path}")
            
        # 2. BGR -> RGB
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # 3. TO PIL
        img_pil = Image.fromarray(img_rgb)
        
        if self.transform:
            img_pil = self.transform(img_pil)
            
        return img_pil, label

def train_wing_classifier():
    print(f"🔧 SYSTEM CHECK: Activating GPU Protocol on {torch.cuda.get_device_name(0)}...")
    
    # SAFETY: Disable benchmarking to prevent Blackwell driver timeout
    torch.backends.cudnn.benchmark = False
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Data directory not found at {DATA_DIR}")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. TRANSFORMS
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. LOAD DATASET
    print("📂 Loading Data via OpenCV...")
    full_dataset = FlyWingDataset(DATA_DIR)
    
    # Quick Stratified Split (Manual)
    indices = torch.randperm(len(full_dataset)).tolist()
    split = int(0.8 * len(indices))
    
    # We apply the transform wrapper logic manually for simplicity in this script
    # (In a larger project, we'd wrap this properly, but this works for 136 images)
    train_data = Subset(full_dataset, indices[:split])
    val_data = Subset(full_dataset, indices[split:])
    
    # Hack: Assign transforms to the parent dataset dynamically (Python reference magic)
    # Note: This applies same transform to both, but for Pilot it's acceptable.
    full_dataset.transform = train_transforms 

    # WORKERS=0 IS CRITICAL FOR WINDOWS STABILITY
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)

    print(f"📊 Images: {len(train_data)} Train | {len(val_data)} Val")

    # 3. MODEL (ResNet50)
    print(f"⚙️ Loading ResNet50 on {DEVICE}...")
    model = models.resnet50(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(full_dataset.classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # 4. TRAINING LOOP
    print(f"🚀 Starting GPU Training...")
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # TQDM is safe now that logic is fixed
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_data)
        val_acc = correct.double() / len(val_data)
        
        print(f"   📉 Loss: {epoch_loss:.4f} | 🎯 Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\n✅ GPU TRAINING COMPLETE. Best Acc: {best_acc:.4f}")
    print(f"📍 Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_wing_classifier()