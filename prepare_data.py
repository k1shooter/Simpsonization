import os
import glob
from PIL import Image
import config
import shutil

def prepare_data():
    dataset_dir = config.DATASET_ROOT
    print(f"Looking for dataset in {dataset_dir}...")
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} does not exist.")
        # Check if we can find it in the unzip location
        # Maybe the zip structure is different
        parent_dir = os.path.dirname(dataset_dir)
        if os.path.exists(parent_dir):
            print(f"Listing {parent_dir}: {os.listdir(parent_dir)}")
        return

    images = glob.glob(os.path.join(dataset_dir, "*.png"))
    if not images:
        # Try recursive
        images = glob.glob(os.path.join(dataset_dir, "**", "*.png"), recursive=True)
        
    if not images:
        print("No images found!")
        return
        
    print(f"Found {len(images)} images.")
    
    # Sort to ensure determinism
    images.sort()
    
    if len(images) <= config.EXEMPLAR_INDEX:
        print(f"Not enough images for index {config.EXEMPLAR_INDEX}")
        return
        
    exemplar_path = images[config.EXEMPLAR_INDEX]
    print(f"Selected Exemplar: {exemplar_path}")
    
    shutil.copy(exemplar_path, "exemplar.png")
    print("Saved exemplar.png")

if __name__ == "__main__":
    prepare_data()
