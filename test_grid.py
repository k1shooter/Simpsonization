import os
import glob
import torch
from PIL import Image, ImageDraw, ImageFont
import config
from inpainting_pipeline import CustomInpaintingPipeline
from tqdm import tqdm
import random

# --- 테스트할 파라미터 범위 설정 ---
# 3D 느낌을 원하시므로 높은 Scale 위주로, 구조 유지를 위해 Strength를 다양하게 테스트
TEST_GUIDANCE_SCALES = [7.5, 12.0, 15.0, 20.0]
TEST_STRENGTHS = [0.6, 0.7, 0.8, 0.9]

# 테스트할 이미지 개수
NUM_TEST_SAMPLES = 100 

def find_mask_path(img_path):
    """이미지 경로를 기반으로 YOLO 라벨 파일 경로를 찾습니다."""
    # .../images/train/abc.jpg -> .../labels/train/abc.txt
    base_dir = os.path.dirname(os.path.dirname(img_path)) # images 상위 폴더
    file_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # 1. 표준 데이터셋 구조 (labels/train/...) 시도
    label_path = img_path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
    
    if not os.path.exists(label_path):
        # 2. 같은 폴더 내 검색 (Fallback)
        label_path = os.path.join(os.path.dirname(img_path), file_name + ".txt")
        
    return label_path if os.path.exists(label_path) else None

def create_grid(images, rows, cols, size=512, row_labels=None, col_labels=None):
    """이미지 리스트를 격자로 합칩니다."""
    w, h = size, size
    grid_w = cols * w
    grid_h = rows * h
    
    # 라벨을 위한 여백 추가 (상단, 좌측)
    margin_top = 50 if col_labels else 0
    margin_left = 100 if row_labels else 0
    
    grid_img = Image.new('RGB', (grid_w + margin_left, grid_h + margin_top), color='white')
    draw = ImageDraw.Draw(grid_img)
    
    # 컬럼 라벨 (Guidance Scale)
    if col_labels:
        for i, label in enumerate(col_labels):
            draw.text((margin_left + i * w + w//2 - 20, 15), str(label), fill="black")
            
    # 행 라벨 (Strength)
    if row_labels:
        for i, label in enumerate(row_labels):
            draw.text((10, margin_top + i * h + h//2), str(label), fill="black")

    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        img = img.resize((w, h))
        grid_img.paste(img, (margin_left + c * w, margin_top + r * h))
        
    return grid_img

def main():
    # 1. 파이프라인 로드 (한 번만 로드)
    pipeline = CustomInpaintingPipeline(config.MODEL_ID, config.OUTPUT_DIR, device=config.DEVICE)
    
    # 2. 테스트 이미지 찾기
    # config.CONTEXT_MASK_ROOT 내의 이미지를 검색
    search_path = os.path.join(config.CONTEXT_MASK_ROOT, "**", "*.jpg")
    all_images = glob.glob(search_path, recursive=True)
    
    if not all_images:
        print("No images found for testing.")
        return

    # 랜덤 샘플링
    test_images = random.sample(all_images, min(len(all_images), NUM_TEST_SAMPLES))
    print(f"Testing on {len(test_images)} images...")

    # 결과 저장 폴더
    save_dir = "test_resultssss"
    os.makedirs(save_dir, exist_ok=True)

    # 3. 각 이미지에 대해 파라미터 그리드 테스트
    for img_idx, img_path in enumerate(test_images):
        mask_path = find_mask_path(img_path)
        if not mask_path:
            print(f"Skipping {img_path}: Mask not found.")
            continue
            
        print(f"\n[{img_idx+1}/{len(test_images)}] Processing: {os.path.basename(img_path)}")
        
        orig_image = Image.open(img_path).convert("RGB")
        results = []
        
        # 중첩 루프: Strength(행) x Guidance(열)
        for strength in TEST_STRENGTHS:
            for guidance in TEST_GUIDANCE_SCALES:
                print(f"  > Testing: Str={strength}, CFG={guidance}")
                
                res = pipeline.inpaint(
                    image=orig_image,
                    mask_path=mask_path,
                    prompt=config.EXEMPLAR_PROMPT,
                    num_inference_steps=config.NUM_INFERENCE_STEPS,
                    guidance_scale=guidance,
                    strength=strength
                )
                results.append(res)
        
        # 그리드 생성 및 저장
        grid = create_grid(
            results, 
            rows=len(TEST_STRENGTHS), 
            cols=len(TEST_GUIDANCE_SCALES), 
            row_labels=[f"Str {s}" for s in TEST_STRENGTHS],
            col_labels=[f"CFG {g}" for g in TEST_GUIDANCE_SCALES]
        )
        
        filename = os.path.basename(img_path).split('.')[0]
        save_path = os.path.join(save_dir, f"grid_{filename}.png")
        grid.save(save_path)
        print(f"Saved grid to {save_path}")

if __name__ == "__main__":
    main()