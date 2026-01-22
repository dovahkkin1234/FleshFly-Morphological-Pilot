import cv2
import os
import numpy as np
from tqdm import tqdm

# --- ROBUST PATH SETUP ---
try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
except NameError:
    PROJECT_ROOT = os.getcwd()

# Data is read from raw
RAW_DIR = os.path.join(PROJECT_ROOT, 'Data', 'rawData')
# Modified data is stored in processed2 to distinguish from the "failure" version
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'Data', 'processed2')

def apply_clahe_enhancement(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    
    # 1. INITIAL CLEANING: Median Blur to remove physical debris (dust/specks)
    # This addresses the "dirty slide" issue before contrast is boosted
    cleaned = cv2.medianBlur(img, 3)
    
    # 2. Convert to LAB color space
    lab = cv2.cvtColor(cleaned, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 3. CLAHE (Local Contrast)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    
    # 4. Merge back
    enhanced_img = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)
    
    # 5. FINAL REFINEMENT: Bilateral Filter
    # Smooths smudge marks while keeping vein edges perfectly sharp
    final_img = cv2.bilateralFilter(enhanced_img, d=5, sigmaColor=50, sigmaSpace=50)
    
    return final_img

def run_preprocessing():
    print(f"🚀 Project Root: {PROJECT_ROOT}")
    print(f"📂 Reading Raw Data: {RAW_DIR}")
    print(f"📂 Saving Improved Data: {PROCESSED_DIR}")
    
    if not os.path.exists(RAW_DIR):
        print(f"❌ Error: Could not find {RAW_DIR}")
        return

    species_list = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    
    for species in species_list:
        species_in = os.path.join(RAW_DIR, species)
        species_out = os.path.join(PROCESSED_DIR, species)
        os.makedirs(species_out, exist_ok=True)
        
        images = [f for f in os.listdir(species_in) if f.lower().endswith(('.tif', '.jpg', '.png'))]
        
        for img_name in tqdm(images, desc=f"Processing {species}"):
            in_file = os.path.join(species_in, img_name)
            out_file = os.path.join(species_out, img_name)
            
            enhanced = apply_clahe_enhancement(in_file)
            if enhanced is not None:
                cv2.imwrite(out_file, enhanced)

if __name__ == "__main__":
    run_preprocessing()
    print(f"\n✅ Success! Improved data is now in: {PROCESSED_DIR}")