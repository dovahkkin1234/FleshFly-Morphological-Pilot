import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- ROBUST PATH SETUP ---
# Detects the project root regardless of whether you run from root or src/
try:
    # If running as a .py script
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
except NameError:
    # If running in a notebook cell
    PROJECT_ROOT = os.getcwd()

# Point to your specific folders
# Note: Using your folder name 'Data' as seen in your terminal output
RAW_DIR = os.path.join(PROJECT_ROOT, 'Data', 'rawData')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'Data', 'processed')

def apply_clahe_enhancement(image_path):
    """
    Enhances wing veins using CLAHE. This makes the background color 
    irrelevant and forces the model to focus on the vein 'texture'.
    """
    img = cv2.imread(image_path)
    if img is None: return None
    
    # Convert to LAB color space (L=Lightness)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    
    # Merge back and convert to BGR
    enhanced_img = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)
    return enhanced_img

def run_preprocessing():
    print(f"🚀 Project Root: {PROJECT_ROOT}")
    print(f"📂 Looking for raw data in: {RAW_DIR}")
    
    if not os.path.exists(RAW_DIR):
        print(f"❌ Error: Could not find {RAW_DIR}")
        return

    # Iterate through species folders
    species_list = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    
    for species in species_list:
        species_in = os.path.join(RAW_DIR, species)
        species_out = os.path.join(PROCESSED_DIR, species)
        os.makedirs(species_out, exist_ok=True)
        
        images = [f for f in os.listdir(species_in) if f.lower().endswith(('.tif', '.jpg', '.png'))]
        
        # tqdm creates the progress bar in your terminal
        for img_name in tqdm(images, desc=f"Processing {species}"):
            in_file = os.path.join(species_in, img_name)
            out_file = os.path.join(species_out, img_name)
            
            enhanced = apply_clahe_enhancement(in_file)
            if enhanced is not None:
                cv2.imwrite(out_file, enhanced)

if __name__ == "__main__":
    run_preprocessing()
    print(f"\n✅ Success! Clean data is now in: {PROCESSED_DIR}")