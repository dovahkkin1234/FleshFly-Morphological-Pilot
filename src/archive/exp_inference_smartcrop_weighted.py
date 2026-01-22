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

# TRUST WEIGHTS (Critical Tuning)
# We give EfficientNet 80% authority because it proved 80% accurate on raw data vs ResNet's 60%.
W_RESNET = 0.2
W_EFFNET = 0.8

try:
    CLASSES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
except:
    CLASSES = ['A.gressitti', 'B.karnyi', 'C.megacephala', 'C.nigripes', 
               'C.rufifacies', 'L.alba', 'S.aquila', 'S.princeps']

def advanced_preprocess(image_path):
    """ CLAHE (Lighting Fix) - The winner from Script 4 """
    img = cv2.imread(image_path)
    if img is None: return None

    # Fix lighting using LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def smart_crop(img):
    """ 
    The 'Smart Crop' from Script 4 (The best performing one).
    It crops the wing but DOES NOT erase the background pixels.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    valid_contours = [c for c in contours if cv2.contourArea(c) > 1000]
    if not valid_contours: valid_contours = contours
    c = max(valid_contours, key=cv2.contourArea)
    
    x, y, w, h = cv2.boundingRect(c)
    crop = img[y:y+h, x:x+w]
    
    # Pad to square (White background padding only)
    h_c, w_c = crop.shape[:2]
    size = max(h_c, w_c) + 50
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
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

def get_prediction_data(img_pil, model, transform):
    """ Returns name, confidence, and full probability vector using TTA """
    # 1. Standard Prediction
    input_normal = transform(img_pil).unsqueeze(0).to(DEVICE)
    
    # 2. Flipped Prediction (Test Time Augmentation)
    input_flipped = transform(ImageOps.mirror(img_pil)).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        out_normal = torch.nn.functional.softmax(model(input_normal)[0], dim=0)
        out_flipped = torch.nn.functional.softmax(model(input_flipped)[0], dim=0)
    
    # Average the two views
    avg_probs = (out_normal + out_flipped) / 2.0
    conf, idx = torch.topk(avg_probs, 1)
    
    return CLASSES[idx.item()], conf.item() * 100, avg_probs

def run_prediction(image_path, resnet, effnet):
    print(f"\n🔍 Processing: {os.path.basename(image_path)}")
    
    try:
        # Step 1: CLAHE + Smart Crop (Preserves Texture)
        enhanced_cv = advanced_preprocess(image_path)
        img_pil = smart_crop(enhanced_cv)
        img_pil.save(os.path.join(PROJECT_ROOT, "debug_best_crop.jpg"))
    except Exception as e:
        print(f"   ⚠️ Preprocessing Failed: {e}")
        return

    # Transforms
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
    
    # Step 2: Get Predictions & Probabilities
    name_r, conf_r, probs_r = get_prediction_data(img_pil, resnet, tx_res)
    name_e, conf_e, probs_e = get_prediction_data(img_pil, effnet, tx_eff)
    
    # Step 3: Weighted Ensemble Fusion
    # This combines the 'confidence vectors' of both models
    final_probs = (probs_r * W_RESNET) + (probs_e * W_EFFNET)
    conf_final, idx_final = torch.topk(final_probs, 1)
    winner = CLASSES[idx_final.item()]
    
    # Step 4: Report Card
    print("-" * 50)
    # Icon logic for visual scanning
    r_icon = "✅" if name_r == "C.nigripes" else "❌" # Change target as needed for testing
    e_icon = "✅" if name_e == "C.nigripes" else "❌"
    
    print(f"   🤖 ResNet50:      {r_icon} {name_r} ({conf_r:.1f}%)")
    print(f"   🤖 EfficientNet:  {e_icon} {name_e} ({conf_e:.1f}%)")
    print("-" * 50)
    
    # Final Decision
    w_icon = "✅" if winner == "C.nigripes" else "❌"
    print(f"   🏆 FINAL DECISION: {w_icon} {winner} ({conf_final.item()*100:.1f}%)")
    
    if name_r != name_e:
        print("   ⚠️ Models disagreed. Weighted vote applied.")
    else:
        print("   ✅ Consensus reached.")

if __name__ == "__main__":
    r_model, e_model = load_models()
    print("\n👉 Paste path to RAW image (or 'q' to quit)")
    while True:
        p = input("\n📂 Path: ").strip().replace('"', '')
        if p.lower() in ['q', 'exit']: break
        if os.path.exists(p):
            run_prediction(p, r_model, e_model)
        else:
            print("❌ File not found.")