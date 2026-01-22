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

# MODEL PATHS
RESNET_PATH = os.path.join(PROJECT_ROOT, 'models', 'wing_classifier_resnet_gpu.pth')
EFFNET_PATH = os.path.join(PROJECT_ROOT, 'models', 'wing_classifier_effnet.pth')
CONVNEXT_PATH = os.path.join(PROJECT_ROOT, 'models', 'wing_classifier_convnext.pth')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- VOTING WEIGHTS (The Trinity) ---
# ResNet is the weakest on raw data, so it gets less say.
# EfficientNet and ConvNeXt are the heavy hitters.
W_RESNET = 0.2
W_EFFNET = 0.4
W_CONVNEXT = 0.4

try:
    CLASSES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
except:
    CLASSES = ['A.gressitti', 'B.karnyi', 'C.megacephala', 'C.nigripes', 
               'C.rufifacies', 'L.alba', 'S.aquila', 'S.princeps']

def advanced_preprocess(image_path):
    """ CLAHE (Lighting Fix) """
    img = cv2.imread(image_path)
    if img is None: return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def smart_crop(img):
    """ The 'Gold Standard' Crop logic """
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
    
    h_c, w_c = crop.shape[:2]
    size = max(h_c, w_c) + 50
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
    y_off = (size - h_c) // 2
    x_off = (size - w_c) // 2
    canvas[y_off:y_off+h_c, x_off:x_off+w_c] = crop
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

def load_models():
    print("⏳ Loading The Trinity...")
    
    # 1. ResNet50
    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, len(CLASSES))
    resnet.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
    resnet.to(DEVICE).eval()
    
    # 2. EfficientNetV2-S
    effnet = models.efficientnet_v2_s(weights=None)
    effnet.classifier[1] = nn.Linear(effnet.classifier[1].in_features, len(CLASSES))
    effnet.load_state_dict(torch.load(EFFNET_PATH, map_location=DEVICE))
    effnet.to(DEVICE).eval()

    # 3. ConvNeXt-Tiny
    convnext = models.convnext_tiny(weights=None)
    convnext.classifier[2] = nn.Linear(convnext.classifier[2].in_features, len(CLASSES))
    convnext.load_state_dict(torch.load(CONVNEXT_PATH, map_location=DEVICE))
    convnext.to(DEVICE).eval()
    
    return resnet, effnet, convnext

def get_prediction_data(img_pil, model, transform):
    input_normal = transform(img_pil).unsqueeze(0).to(DEVICE)
    input_flipped = transform(ImageOps.mirror(img_pil)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out_n = torch.nn.functional.softmax(model(input_normal)[0], dim=0)
        out_f = torch.nn.functional.softmax(model(input_flipped)[0], dim=0)
    avg_probs = (out_n + out_f) / 2.0
    conf, idx = torch.topk(avg_probs, 1)
    return CLASSES[idx.item()], conf.item() * 100, avg_probs

def run_prediction(image_path, r_model, e_model, c_model):
    print(f"\n🔍 Processing: {os.path.basename(image_path)}")
    try:
        enhanced_cv = advanced_preprocess(image_path)
        img_pil = smart_crop(enhanced_cv)
        img_pil.save(os.path.join(PROJECT_ROOT, "debug_trinity_crop.jpg"))
    except Exception as e:
        print(f"   ⚠️ Preprocessing Failed: {e}")
        return

    # Transforms
    tx_224 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tx_384 = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Get Individual Predictions
    name_r, conf_r, probs_r = get_prediction_data(img_pil, r_model, tx_224)
    name_e, conf_e, probs_e = get_prediction_data(img_pil, e_model, tx_384)
    name_c, conf_c, probs_c = get_prediction_data(img_pil, c_model, tx_224)
    
    # --- TRINITY FUSION ---
    final_probs = (probs_r * W_RESNET) + (probs_e * W_EFFNET) + (probs_c * W_CONVNEXT)
    conf_final, idx_final = torch.topk(final_probs, 1)
    winner = CLASSES[idx_final.item()]
    
    # Report
    print("-" * 60)
    print(f"   🤖 ResNet50 (20%):      {name_r} ({conf_r:.1f}%)")
    print(f"   🤖 EfficientNet (40%):  {name_e} ({conf_e:.1f}%)")
    print(f"   🤖 ConvNeXt (40%):      {name_c} ({conf_c:.1f}%)")
    print("-" * 60)
    
    icon = "✅" if winner == "C.nigripes" else "❌" # Target check
    print(f"   🏆 TRINITY CONSENSUS: {icon} {winner} ({conf_final.item()*100:.1f}%)")
    
    if name_r == name_e == name_c:
        print("   ✅ Unanimous Agreement.")
    elif name_e == name_c:
        print("   ✅ Strong Consensus (EffNet + ConvNeXt).")
    else:
        print("   ⚠️ Split Decision. Weighted vote applied.")

if __name__ == "__main__":
    r, e, c = load_models()
    print("\n👉 Paste path to RAW image (or 'q' to quit)")
    while True:
        p = input("\n📂 Path: ").strip().replace('"', '')
        if p.lower() in ['q', 'exit']: break
        if os.path.exists(p):
            run_prediction(p, r, e, c)
        else:
            print("❌ File not found.")