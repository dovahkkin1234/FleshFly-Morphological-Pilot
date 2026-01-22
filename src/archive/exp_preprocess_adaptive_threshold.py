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
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'Data', 'processed7')

def final_safe_extraction(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    
    # 1. SIMPLEST PATH: Convert to gray and use Adaptive Thresholding
    # This identifies the "mass" of the wing rather than just the edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Use Otsu to find the best separation between wing and slide
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. CLEANUP: Close gaps in the wing structure
    kernel = np.ones((15,15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 3. SELECT WING: Keep only the largest continuous object
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 4. TIGHT CROP: Get the bounding box of the wing
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Apply your requested safety margins (Extra room for the bottom and root)
    pad_top, pad_right = 20, 20
    pad_bottom, pad_left = 120, 100 
    
    y1 = max(0, y - pad_top)
    y2 = min(img.shape[0], y + h + pad_bottom)
    x1 = max(0, x - pad_left)
    x2 = min(img.shape[1], x + w + pad_right)
    
    # 5. FINAL RESULT: Original pixels, no "destruction"
    cropped = img[y1:y2, x1:x2]
    
    # Mild CLAHE just to make the veins pop for the model
    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

# --- RUNNING LOGIC ---
def run_final():
    print(f"🚀 Running Final Safe Pipeline...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    species_list = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    
    for species in species_list:
        s_out = os.path.join(PROCESSED_DIR, species)
        os.makedirs(s_out, exist_ok=True)
        images = [f for f in os.listdir(os.path.join(RAW_DIR, species)) if f.lower().endswith(('.tif', '.png', '.jpg'))]
        
        for img_name in tqdm(images, desc=f"Cropping {species}"):
            res = final_safe_extraction(os.path.join(RAW_DIR, species, img_name))
            if res is not None:
                cv2.imwrite(os.path.join(s_out, img_name), res)

if __name__ == "__main__":
    run_final()