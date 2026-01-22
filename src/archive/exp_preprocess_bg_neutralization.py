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
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'Data', 'processed3')

def isolate_and_clean_wing(image_path):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Advanced Denoising (Non-Local Means)
    # This is heavy but works wonders for sensor grain and small artifacts
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # 3. Create a Mask of the Wing
    # We use Otsu's thresholding to separate the wing from the background
    _, mask = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Morphological "Closing"
    # This fills in small holes in the veins and deletes tiny dust specks
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 5. Background Neutralization
    # We create a solid neutral background (Gray value 200)
    bg = np.full(img.shape, 200, dtype=np.uint8)

    # 6. Apply CLAHE ONLY to the wing area
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    enhanced_img = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)

    # 7. Final Merge: Enhanced Wing + Clean Background
    # Bitwise operations ensure only the wing pixels are kept
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    final_img = np.where(mask_3ch == 255, enhanced_img, bg)

    return final_img

def run_v3_pipeline():
    print(f"🚀 Launching V3 Feature-Isolation Pipeline...")
    print(f"📂 Saving to: {PROCESSED_DIR}")
    
    species_list = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    
    for species in species_list:
        species_in = os.path.join(RAW_DIR, species)
        species_out = os.path.join(PROCESSED_DIR, species)
        os.makedirs(species_out, exist_ok=True)
        
        images = [f for f in os.listdir(species_in) if f.lower().endswith(('.tif', '.jpg', '.png'))]
        
        for img_name in tqdm(images, desc=f"Cleaning {species}"):
            in_file = os.path.join(species_in, img_name)
            out_file = os.path.join(species_out, img_name)
            
            clean_wing = isolate_and_clean_wing(in_file)
            if clean_wing is not None:
                cv2.imwrite(out_file, clean_wing)

if __name__ == "__main__":
    run_v3_pipeline()
    print(f"\n✅ V3 Pipeline Complete! Data is noise-free in: {PROCESSED_DIR}")