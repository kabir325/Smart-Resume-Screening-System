import os
import kaggle
from tqdm import tqdm

# Ensure API key is set up
DATASET_NAME = "gauravduttakiit/resume-dataset"
SAVE_PATH = "dataset"

os.makedirs(SAVE_PATH, exist_ok=True)

print("📥 Downloading dataset...")
kaggle.api.dataset_download_files(DATASET_NAME, path=SAVE_PATH, unzip=True)

# Show progress
for _ in tqdm(range(100), desc="Extracting Files"):
    pass  # Simulating progress

print(f"✅ Dataset downloaded to '{SAVE_PATH}'")
