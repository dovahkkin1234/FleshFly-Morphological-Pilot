import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import cv2
import numpy as np

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data', 'processed8')

# MODEL PATHS
RESNET_PATH = os.path.join(PROJECT_ROOT, 'models', 'wing_classifier_resnet_gpu.pth')
EFFNET_PATH = os.path.join(PROJECT_ROOT, 'models', 'wing_classifier_effnet.pth')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GET CLASS NAMES AUTOMATICALLY
try:
    CLASSES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
except FileNotFoundError:
    # Fallback if Data folder is moved
    CLASSES = ['A.gressitti', 'B.karnyi', 'C.megacephala', 'C.nigripes', 
               'C.rufifacies', 'L.alba', 'S.aquila', 'S.princeps']

def load_models():
    print(f"⏳ Loading AI Models on {DEVICE}...")
    
    # Load ResNet50
    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, len(CLASSES))
    resnet.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
    resnet.to(DEVICE)
    resnet.eval()
    
    # Load EfficientNetV2-S
    effnet = models.efficientnet_v2_s(weights=None)
    effnet.classifier[1] = nn.Linear(effnet.classifier[1].in_features, len(CLASSES))
    effnet.load_state_dict(torch.load(EFFNET_PATH, map_location=DEVICE))
    effnet.to(DEVICE)
    effnet.eval()
    
    print("✅ Models Loaded & Ready.")
    return resnet, effnet

def preprocess_image(image_path, target_size):
    # Use OpenCV to handle complex formats/metadata safely
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise ValueError(f"Could not open image at: {image_path}")
        
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return transform(img_pil).unsqueeze(0).to(DEVICE)

def predict(image_path, resnet, effnet):
    print(f"\n🔍 Analyzing: {os.path.basename(image_path)} ...")
    
    try:
        # 1. ResNet Prediction (224x224)
        input_res = preprocess_image(image_path, 224)
        with torch.no_grad():
            out_res = resnet(input_res)
            prob_res = torch.nn.functional.softmax(out_res[0], dim=0)
            conf_res, idx_res = torch.topk(prob_res, 1)
            
        # 2. EfficientNet Prediction (384x384)
        input_eff = preprocess_image(image_path, 384)
        with torch.no_grad():
            out_eff = effnet(input_eff)
            prob_eff = torch.nn.functional.softmax(out_eff[0], dim=0)
            conf_eff, idx_eff = torch.topk(prob_eff, 1)

        name_res = CLASSES[idx_res.item()]
        name_eff = CLASSES[idx_eff.item()]
        score_res = conf_res.item() * 100
        score_eff = conf_eff.item() * 100

        print(f"   🤖 ResNet50:      {name_res} ({score_res:.1f}%)")
        print(f"   🤖 EfficientNet:  {name_eff} ({score_eff:.1f}%)")
        print("-" * 40)
        
        if name_res == name_eff:
            print(f"   ✅ FINAL RESULT: {name_res}")
        else:
            print(f"   ⚠️ CONFLICT: Models disagree. Manual review recommended.")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    resnet_model, effnet_model = load_models()
    
    print("\n" + "="*50)
    print("      FLESH FLY CLASSIFIER - INTERACTIVE MODE")
    print("="*50)
    print("👉 To test an image, right-click the file in Windows Explorer,")
    print("   select 'Copy as path', and paste it here.")
    print("👉 Type 'exit' to quit.\n")

    while True:
        user_input = input("📂 Paste Image Path: ").strip()
        
        # Remove quotes if Windows adds them ("C:\Path\...")
        user_input = user_input.replace('"', '')
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
            
        if not os.path.exists(user_input):
            print("   ❌ File not found. Try again.")
            continue
            
        predict(user_input, resnet_model, effnet_model)