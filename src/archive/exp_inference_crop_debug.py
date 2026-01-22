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

# Paths to your trained models
RESNET_PATH = os.path.join(PROJECT_ROOT, 'models', 'wing_classifier_resnet_gpu.pth')
EFFNET_PATH = os.path.join(PROJECT_ROOT, 'models', 'wing_classifier_effnet.pth')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GET CLASS NAMES AUTOMATICALLY
try:
    CLASSES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
except FileNotFoundError:
    # Fallback default classes if data folder is missing
    CLASSES = ['A.gressitti', 'B.karnyi', 'C.megacephala', 'C.nigripes', 
               'C.rufifacies', 'L.alba', 'S.aquila', 'S.princeps']

def auto_crop_wing(image_path):
    """
    Simulates the training preprocessing:
    1. Thresholds the image to find the dark wing.
    2. Crops to the largest contour.
    3. Pastes it onto a clean white square background.
    """
    img = cv2.imread(image_path)
    if img is None: return None

    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Threshold (Invert because wings are usually dark on light bg)
    # Adjust '200' if your images are darker/lighter
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 3. Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # If no contours found, return original image converted to PIL
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 4. Get largest contour (The Wing)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # 5. Crop the wing
    crop = img[y:y+h, x:x+w]
    
    # 6. Paste onto square white background (Padding)
    h_c, w_c = crop.shape[:2]
    size = max(h_c, w_c) + 50  # Add 50px padding
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255 # White canvas
    
    # Center the crop
    y_off = (size - h_c) // 2
    x_off = (size - w_c) // 2
    canvas[y_off:y_off+h_c, x_off:x_off+w_c] = crop
    
    # Convert to PIL for PyTorch
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

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
    
    print("✅ Models Loaded.")
    return resnet, effnet

def predict(image_path, resnet, effnet):
    print(f"\n🔍 Processing: {os.path.basename(image_path)}")
    
    # STEP 1: AUTO-CROP & DEBUG SAVE
    try:
        img_pil = auto_crop_wing(image_path)
        
        # --- DEBUG FEATURE ---
        # Save what the AI actually sees to "debug_crop.jpg" in the project root
        debug_path = os.path.join(PROJECT_ROOT, "debug_crop.jpg")
        img_pil.save(debug_path)
        # ---------------------
        
    except Exception as e:
        print(f"   ⚠️ Crop failed (Using raw image): {e}")
        img_cv = cv2.imread(image_path)
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    # STEP 2: PREPARE TENSORS
    # ResNet expects 224x224
    tx_res = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # EfficientNet expects 384x384
    tx_eff = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_res = tx_res(img_pil).unsqueeze(0).to(DEVICE)
    input_eff = tx_eff(img_pil).unsqueeze(0).to(DEVICE)

    # STEP 3: PREDICT
    with torch.no_grad():
        # ResNet Inference
        out_res = resnet(input_res)
        prob_res = torch.nn.functional.softmax(out_res[0], dim=0)
        conf_res, idx_res = torch.topk(prob_res, 1)
        
        # EfficientNet Inference
        out_eff = effnet(input_eff)
        prob_eff = torch.nn.functional.softmax(out_eff[0], dim=0)
        conf_eff, idx_eff = torch.topk(prob_eff, 1)

    name_res = CLASSES[idx_res.item()]
    name_eff = CLASSES[idx_eff.item()]
    
    # Visual Feedback
    print(f"   🤖 ResNet50:      {name_res} ({conf_res.item()*100:.1f}%)")
    print(f"   🤖 EfficientNet:  {name_eff} ({conf_eff.item()*100:.1f}%)")
    
    if name_res == name_eff:
         print(f"   ✅ CONSENSUS: {name_res}")
    else:
         print(f"   ⚠️ CONFLICT: Models disagree.")
         
    print(f"   📸 Debug crop saved to: {os.path.join(PROJECT_ROOT, 'debug_crop.jpg')}")

if __name__ == "__main__":
    resnet_model, effnet_model = load_models()
    
    print("\n" + "="*50)
    print("      AUTO-CROPPING INFERENCE ENGINE")
    print("="*50)
    print("👉 Check 'debug_crop.jpg' after each run to see the crop.")
    
    while True:
        user_input = input("\n📂 Paste Image Path (or 'q' to quit): ").strip().replace('"', '')
        
        if user_input.lower() in ['exit', 'q', 'quit']:
            break
            
        if os.path.exists(user_input):
            predict(user_input, resnet_model, effnet_model)
        else:
            print("❌ File not found. Try again.")