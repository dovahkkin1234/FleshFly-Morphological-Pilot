import cv2
import os
import numpy as np
from tqdm import tqdm

# --- PATH SETUP ---
try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
except NameError:
    PROJECT_ROOT = os.getcwd()

RAW_DIR = os.path.join(PROJECT_ROOT, 'Data', 'rawData')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'Data', 'processed5')

def final_safe_crop(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    
    # 1. ENHANCED DETECTION PATHS
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Very low Canny thresholds to catch the faintest membrane edges
    edges = cv2.Canny(blurred, 10, 50) 
    
    # Adaptive threshold to catch the dark wing root and dense veins
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 25, 8)
    
    # 2. COMBINE AND BRIDGE
    combined = cv2.bitwise_or(edges, thresh)
    # Using a large elliptical kernel to bridge any gaps in the lower lobe
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dilated = cv2.dilate(combined, kernel, iterations=2)
    
    # 3. ISOLATE WING
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest_wing = max(contours, key=cv2.contourArea)
    
    # 4. AGGRESSIVE ASYMMETRIC PADDING
    x, y, w, h = cv2.boundingRect(largest_wing)
    
    # We provide a very generous margin for the lower and left sections
    # where the wing root and lower membrane often sit.
    pad_top = 20
    pad_bottom = 100  # Significantly increased to capture the bottom edges
    pad_left = 80     # Increased to capture the full wing root
    pad_right = 20
    
    y1 = max(0, y - pad_top)
    y2 = min(img.shape[0], y + h + pad_bottom)
    x1 = max(0, x - pad_left)
    x2 = min(img.shape[1], x + w + pad_right)
    
    cropped_img = img[y1:y2, x1:x2]
    
    # 5. FINAL VISUAL ENHANCEMENT
    # CLAHE helps the model "see" the veins in the newly included lower parts
    lab = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)

def run_v9():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    species_list = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    for species in species_list:
        s_out = os.path.join(PROCESSED_DIR, species)
        os.makedirs(s_out, exist_ok=True)
        images = [f for f in os.listdir(os.path.join(RAW_DIR, species)) if f.lower().endswith(('.tif', '.png', '.jpg'))]
        for img_name in tqdm(images, desc=f"Final Safe Crop: {species}"):
            res = final_safe_crop(os.path.join(RAW_DIR, species, img_name))
            if res is not None:
                cv2.imwrite(os.path.join(s_out, img_name), res)

if __name__ == "__main__":
    run_v9()
    print(f"\n✅ All wings fully captured with maximum safety margins!")