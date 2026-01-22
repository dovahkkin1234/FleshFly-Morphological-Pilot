import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- PATH SETUP ---
# Get the location of this script (src/eda_viz.py)
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to project_root, then down to data/raw
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data', 'rawData')

print(f"📍 Looking for data at: {DATA_DIR}")

if not os.path.exists(DATA_DIR):
    print(f"❌ Error: Directory not found. Please ensure your images are in '{DATA_DIR}'")
    exit()

# --- COUNT SPECIES ---
species_counts = {}
for species in os.listdir(DATA_DIR):
    species_path = os.path.join(DATA_DIR, species)
    if os.path.isdir(species_path):
        # Count only valid image files
        count = len([f for f in os.listdir(species_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        species_counts[species] = count

print(f"✅ Found {len(species_counts)} species classes.")

# --- PLOT & SAVE ---
plt.figure(figsize=(10, 6))
# Updated to fix the Seaborn warning
sns.barplot(x=list(species_counts.keys()), y=list(species_counts.values()), hue=list(species_counts.keys()), legend=False, palette='viridis')

plt.title('Species Distribution in Flesh Fly Dataset', fontsize=15)
plt.ylabel('Number of Images', fontsize=12)
plt.xticks(rotation=45)

# Save the plot to the project root (not inside src)
output_path = os.path.join(PROJECT_ROOT, 'species_distribution.png')
plt.savefig(output_path, bbox_inches='tight')
print(f"📊 Chart saved to: {output_path}")

plt.show()