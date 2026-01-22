import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATA_DIR = 'Data/processed8'
MODEL_SAVE_PATH = 'models/wing_classifier.pth'
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs('models', exist_ok=True)

# 1. DATA AUGMENTATION
# Helps the model generalize by creating variations of the training images.
train_transforms = transforms.Compose([
    transforms.Resize((384, 384)), # EfficientNetV2-S optimal input size
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. DATA LOADERS
full_dataset = datasets.ImageFolder(DATA_DIR)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Assign specific transforms to subsets
train_data.dataset.transform = train_transforms
val_data.dataset.transform = val_transforms

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# 3. MODEL INITIALIZATION
# Load pre-trained EfficientNetV2-S
model = models.efficientnet_v2_s(weights='DEFAULT')

# Freeze early layers (optional, but good for very small datasets)
# for param in model.parameters():
#     param.requires_grad = False

# Replace the classifier head for the number of species classes
num_classes = len(full_dataset.classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(DEVICE)

# 4. LOSS & OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. TRAINING LOOP
def train():
    print(f"🚀 Training started on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / train_size
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Training complete. Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()