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

# GET CLASS NAMES
try:
    CLASSES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
except:
    CLASSES = ['A.gressitti', 'B.karnyi', 'C.megacephala', 'C.nigripes', 
               'C.rufifacies', 'L.alba', 'S.aquila', 'S.princeps']

def advanced_preprocess(image_path):
    """
    UPGRADE 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Fixes bad lighting before we even try to crop.
    """
    img = cv2.imread(image_path)
    if img is None: return None

    # Convert to LAB color space to isolate 'Lightness'
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # Merge and convert back to BGR
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced_img

def smart_crop(img):
    """
    UPGRADE 2: Otsu's Thresholding & Morphology
    Finds the wing even in noisy backgrounds.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur to remove "speckle" noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu's Thresholding (Auto-calculates the split value)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphology (Close small holes in the wing mask)
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find Contours
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Filter contours by Area (Ignore tiny specks)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 1000]
    if not valid_contours: valid_contours = contours
        
    c = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Crop and Pad
    crop = img[y:y+h, x:x+w]
    
    # Square Padding (White Background)
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

def predict_with_tta(img_pil, model, transform):
    """
    UPGRADE 3: Test-Time Augmentation (TTA)
    Predicts on the original image AND a flipped version, then averages.
    """
    # 1. Original
    input_normal = transform(img_pil).unsqueeze(0).to(DEVICE)
    
    # 2. Flipped
    img_flipped = ImageOps.mirror(img_pil)
    input_flipped = transform(img_flipped).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        out_normal = torch.nn.functional.softmax(model(input_normal)[0], dim=0)
        out_flipped = torch.nn.functional.softmax(model(input_flipped)[0], dim=0)
        
    # Average the confidence scores
    avg_probs = (out_normal + out_flipped) / 2.0
    conf, idx = torch.topk(avg_probs, 1)
    
    return CLASSES[idx.item()], conf.item() * 100

def run_prediction(image_path, resnet, effnet):
    print(f"\n🔍 Processing: {os.path.basename(image_path)}")
    
    try:
        # Step 1: Enhance Lighting
        enhanced_cv = advanced_preprocess(image_path)
        
        # Step 2: Smart Crop
        img_pil = smart_crop(enhanced_cv)
        
        # Debug Save
        debug_path = os.path.join(PROJECT_ROOT, "debug_smart_crop.jpg")
        img_pil.save(debug_path)
        
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
    
    # Step 3: Run TTA Prediction
    name_res, conf_res = predict_with_tta(img_pil, resnet, tx_res)
    name_eff, conf_eff = predict_with_tta(img_pil, effnet, tx_eff)
    
    res_icon = "✅" if name_res == "C.nigripes" else "❌"
    eff_icon = "✅" if name_eff == "C.nigripes" else "❌"

    print(f"   🤖 ResNet50 (TTA):      {res_icon} {name_res} ({conf_res:.1f}%)")
    print(f"   🤖 EfficientNet (TTA):  {eff_icon} {name_eff} ({conf_eff:.1f}%)")
    print(f"   📸 Debug crop saved to: debug_smart_crop.jpg")

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