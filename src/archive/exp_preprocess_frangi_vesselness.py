import cv2
import os
import numpy as np
from skimage.filters import frangi # Requires: pip install scikit-image
from tqdm import tqdm

# --- PATH SETUP ---
try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
except NameError:
    PROJECT_ROOT = os.getcwd()

RAW_DIR = os.path.join(PROJECT_ROOT, 'Data', 'rawData')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'Data', 'processed6')

def advanced_noise_erasure(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    
    # 1. CONVERT & PRE-PROCESS
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Invert because Frangi looks for light structures on dark backgrounds
    inv_gray = cv2.bitwise_not(gray)

    # 2. FRANGI VESSELNESS FILTER
    # This highlights veins (vessel-like) and ignores non-tubular debris
    # scale_range: targets the width of the veins
    # black_ridges: True because our veins are dark on light
    vein_map = frangi(gray, sigmas=range(1, 4, 1), black_ridges=True)
    
    # Normalize the map back to 0-255
    vein_map = cv2.normalize(vein_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. MASKING THE BACKGROUND
    # Use the vein map to create a clean stencil of the wing
    _, mask = cv2.threshold(vein_map, 10, 255, cv2.THRESH_BINARY)
    
    # Remove small isolated objects (the leftover hair bits)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))

    # 4. RECONSTRUCT ORIGINAL PIXELS
    # Keep the original color only where the "Vesselness" was high
    clean_bg = np.full(img.shape, 210, dtype=np.uint8) # Neutral Gray
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Apply CLAHE to the original image first
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced_orig = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Merge: Enhanced wing pixels inside the mask, clean gray outside
    isolated_wing = np.where(mask_3ch == 255, enhanced_orig, clean_bg)

    # 5. ASYMMETRIC CROP
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    pad_bottom, pad_left = 100, 80 # Your requested safety margins
    y1, y2 = max(0, y-10), min(img.shape[0], y+h+pad_bottom)
    x1, x2 = max(0, x-pad_left), min(img.shape[1], x+w+20)

    return isolated_wing[y1:y2, x1:x2]

# (Run logic same as previous versions)