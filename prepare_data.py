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
        return

    images = glob.glob(os.path.join(dataset_dir, "*.png")) + glob.glob(os.path.join(dataset_dir, "*.jpg"))
    
    if not images:
        images = glob.glob(os.path.join(dataset_dir, "**", "*.png"), recursive=True) + \
                 glob.glob(os.path.join(dataset_dir, "**", "*.jpg"), recursive=True)
        
    if not images:
        print("No images found!")
        return
        
    print(f"Found {len(images)} images.")
    images.sort()
    
    if len(images) <= config.EXEMPLAR_INDEX:
        print(f"Not enough images for index {config.EXEMPLAR_INDEX}")
        return
        
    exemplar_path = images[config.EXEMPLAR_INDEX]
    print(f"Selected Exemplar: {exemplar_path}")
    
    # 1. Exemplar 이미지 복사
    shutil.copy(exemplar_path, "exemplar.png")
    print("Saved exemplar.png")
    
    # 2. (주의) Exemplar의 마스크 파일이 없으므로, 로딩 및 복사 로직을 생략합니다.
    #    학습 단계에서 전체 이미지 영역을 마스크로 가정합니다.

if __name__ == "__main__":
    prepare_data()