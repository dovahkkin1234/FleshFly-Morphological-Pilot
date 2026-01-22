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
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'Data', 'processed4')

def get_zero_loss_crop(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    
    # 1. Denoise to ignore hair but keep wing edges
    denoised = cv2.medianBlur(img, 5)
    
    # 2. CREATE MULTI-MASK (The Safety Net)
    # Mask A: Color-based (detects the amber/brown pigment)
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    mask_color = cv2.inRange(hsv, np.array([0, 5, 20]), np.array([50, 255, 255]))
    
    # Mask B: Intensity-based (detects anything darker than the light slide)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    _, mask_intensity = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Combine: If it has color OR it's dark, it's a wing
    combined_mask = cv2.bitwise_or(mask_color, mask_intensity)

    # 3. MOP-UP: Fill internal holes in the wing so the center isn't "lost"
    kernel = np.ones((9,9), np.uint8)
    mask_closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # 4. OBJECT ISOLATION: Keep the largest biological structure
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 5. CONVEX HULL: The "Shrink-Wrap"
    # This ensures even the faint, curved edges of the wing are enclosed
    hull = cv2.convexHull(largest_contour)
    x, y, w, h = cv2.boundingRect(hull)
    
    # 6. CROP WITH SAFETY BUFFER
    # We add 10 pixels of padding to guarantee NO edge is cut off
    pad = 10
    y1, y2 = max(0, y-pad), min(img.shape[0], y+h+pad)
    x1, x2 = max(0, x-pad), min(img.shape[1], x+w+pad)
    
    cropped_img = img[y1:y2, x1:x2]
    
    # 7. FINAL ENHANCEMENT
    lab = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)

def run_v5():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    species_list = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    for species in species_list:
        s_in, s_out = os.path.join(RAW_DIR, species), os.path.join(PROCESSED_DIR, species)
        os.makedirs(s_out, exist_ok=True)
        images = [f for f in os.listdir(s_in) if f.lower().endswith(('.tif', '.jpg', '.png'))]
        for img_name in tqdm(images, desc=f"Safe Cropping {species}"):
            res = get_zero_loss_crop(os.path.join(s_in, img_name))
            if res is not None:
                cv2.imwrite(os.path.join(s_out, img_name), res)

if __name__ == "__main__":
    run_v5()
    print(f"\n✅ Zero-Loss Pipeline Complete! Saved to: {PROCESSED_DIR}")