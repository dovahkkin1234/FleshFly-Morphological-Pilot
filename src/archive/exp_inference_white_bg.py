import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import os
import cv2
import numpy as np

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data', 'processed8')

RESNET_PATH = os.path.join(PROJECT_ROOT, 'models', 'wing_classifier_resnet_gpu.pth')
EFFNET_PATH = os.path.join(PROJECT_ROOT, 'models', 'wing_classifier_effnet.pth')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GET CLASSES
try:
    CLASSES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
except:
    CLASSES = ['A.gressitti', 'B.karnyi', 'C.megacephala', 'C.nigripes', 
               'C.rufifacies', 'L.alba', 'S.aquila', 'S.princeps']

def auto_crop_wing(image_path):
    """
    Automatically detects the wing, crops it, and places it on a white background.
    This simulates the training data for raw inputs.
    """
    img = cv2.imread(image_path)
    if img is None: return None

    # 1. Convert to grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Invert if the background is light (standard microscopy)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 2. Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Return original if fail

    # 3. Get largest contour (The Wing)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # 4. Crop
    crop = img[y:y+h, x:x+w]
    
    # 5. Paste onto square white background (Padding)
    # Create white canvas slightly larger than crop
    h_c, w_c = crop.shape[:2]
    size = max(h_c, w_c) + 50
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
    
    # Center the crop
    y_off = (size - h_c) // 2
    x_off = (size - w_c) // 2
    canvas[y_off:y_off+h_c, x_off:x_off+w_c] = crop
    
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

def load_models():
    print("⏳ Loading Models...")
    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, len(CLASSES))
    resnet.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
    resnet.to(DEVICE)
    resnet.eval()
    
    effnet = models.efficientnet_v2_s(weights=None)
    effnet.classifier[1] = nn.Linear(effnet.classifier[1].in_features, len(CLASSES))
    effnet.load_state_dict(torch.load(EFFNET_PATH, map_location=DEVICE))
    effnet.to(DEVICE)
    effnet.eval()
    return resnet, effnet

def predict(image_path, resnet, effnet):
    print(f"\n🔍 Processing: {os.path.basename(image_path)}")
    
    # STEP 1: AUTO-CROP ( The Magic Fix )
    try:
        img_pil = auto_crop_wing(image_path)
    except Exception as e:
        print(f"   ⚠️ Crop failed, using raw image. Error: {e}")
        img_cv = cv2.imread(image_path)
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    # STEP 2: PREPARE TENSORS
    tx_res = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tx_eff = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_res = tx_res(img_pil).unsqueeze(0).to(DEVICE)
    input_eff = tx_eff(img_pil).unsqueeze(0).to(DEVICE)

    # STEP 3: PREDICT
    with torch.no_grad():
        # ResNet
        out_res = resnet(input_res)
        prob_res = torch.nn.functional.softmax(out_res[0], dim=0)
        conf_res, idx_res = torch.topk(prob_res, 1)
        
        # EffNet
        out_eff = effnet(input_eff)
        prob_eff = torch.nn.functional.softmax(out_eff[0], dim=0)
        conf_eff, idx_eff = torch.topk(prob_eff, 1)

    name_res = CLASSES[idx_res.item()]
    name_eff = CLASSES[idx_eff.item()]
    
    # Color-coded output
    res_icon = "✅" if name_res == "C.nigripes" else "❌"
    eff_icon = "✅" if name_eff == "C.nigripes" else "❌"

    print(f"   🤖 ResNet50:      {res_icon} {name_res} ({conf_res.item()*100:.1f}%)")
    print(f"   🤖 EfficientNet:  {eff_icon} {name_eff} ({conf_eff.item()*100:.1f}%)")

if __name__ == "__main__":
    resnet_model, effnet_model = load_models()
    print("\n👉 Paste the path to your RAW images below.")
    
    while True:
        user_input = input("\n📂 Paste Image Path: ").strip().replace('"', '')
        if user_input.lower() in ['exit', 'q']: break
        if os.path.exists(user_input):
            predict(user_input, resnet_model, effnet_model)
        else:
            print("❌ File not found.")