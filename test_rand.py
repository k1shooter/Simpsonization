import torch
import os
import glob
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm

import config
from inpainting_pipeline import CustomInpaintingPipeline, yolo_to_mask_image

# --- 테스트 설정 ---
NUM_SAMPLES = 100  # 테스트할 이미지 개수
OUTPUT_DIR = "test_results_random2" # 결과 저장 폴더

# 고정 파라미터 (3D 심슨 스타일 최적화)
GUIDANCE_SCALE = 16.0
STRENGTH = 0.8

def find_mask_path(img_path):
    """이미지 경로에 대응하는 YOLO 라벨(.txt) 경로 찾기"""
    # 구조: .../images/train/abc.jpg -> .../labels/train/abc.txt
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    parent_dir = os.path.dirname(img_path) # .../images/train
    
    # 1. 표준 구조 시도 (images -> labels)
    # parent_dir가 'images'를 포함한다고 가정
    if "images" in parent_dir:
        label_dir = parent_dir.replace("images", "labels")
        label_path = os.path.join(label_dir, base_name + ".txt")
        if os.path.exists(label_path):
            return label_path
            
    # 2. 같은 폴더 검색
    label_path = os.path.join(parent_dir, base_name + ".txt")
    if os.path.exists(label_path):
        return label_path
        
    return None

def create_comparison_grid(original, mask, result, filename):
    """[원본 | 마스크 | 결과]를 가로로 이어 붙인 이미지 생성"""
    w, h = original.size
    
    # 마스크 시각화 (RGB 변환)
    if mask.mode != 'RGB':
        mask = mask.convert('RGB')
    
    # 텍스트 라벨 추가를 위한 여백
    header_height = 50
    grid_img = Image.new('RGB', (w * 3, h + header_height), (255, 255, 255))
    draw = ImageDraw.Draw(grid_img)
    
    # 이미지 붙여넣기
    grid_img.paste(original, (0, header_height))
    grid_img.paste(mask, (w, header_height))
    grid_img.paste(result, (w * 2, header_height))
    
    # 라벨 텍스트
    labels = ["Original", "Mask Area", "LoRA Result"]
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()

    for i, label in enumerate(labels):
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        x_pos = (i * w) + (w - text_w) // 2
        draw.text((x_pos, 10), label, fill="black", font=font)
        
    return grid_img

def main():
    # 1. 결과 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"=== Random Sample Testing (N={NUM_SAMPLES}) ===")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # 2. 파이프라인 로드
    pipeline = CustomInpaintingPipeline(
        base_model_id=config.MODEL_ID,
        lora_weights_path=config.OUTPUT_DIR,
        device=config.DEVICE
    )
    
    # 3. 이미지 목록 스캔
    print(f"Scanning images in {config.CONTEXT_MASK_ROOT}...")
    all_images = glob.glob(os.path.join(config.CONTEXT_MASK_ROOT, "**", "*.jpg"), recursive=True)
    all_images += glob.glob(os.path.join(config.CONTEXT_MASK_ROOT, "**", "*.png"), recursive=True)
    
    if not all_images:
        print("No images found! Check config.CONTEXT_MASK_ROOT")
        return
        
    # 4. 랜덤 샘플링
    selected_images = random.sample(all_images, min(len(all_images), NUM_SAMPLES))
    print(f"Selected {len(selected_images)} images for testing.")
    
    # 5. 추론 루프
    for i, img_path in enumerate(tqdm(selected_images)):
        try:
            # 이미지 로드
            original_image = Image.open(img_path).convert("RGB")
            
            # 마스크 로드
            mask_path = find_mask_path(img_path)
            if not mask_path:
                print(f"Skipping {os.path.basename(img_path)}: Mask txt not found.")
                continue
                
            mask_image = yolo_to_mask_image(mask_path, original_image.size)
            
            # 인페인팅 실행
            result_image = pipeline.inpaint(
                image=original_image,
                mask_path=mask_path, # txt 경로 전달 (내부에서 로드)
                prompt=config.EXEMPLAR_PROMPT,
                num_inference_steps=config.NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                strength=STRENGTH
            )
            
            # 비교 이미지 생성 및 저장
            file_name = f"sample_{i+1:02d}_{os.path.basename(img_path)}"
            grid_img = create_comparison_grid(original_image, mask_image, result_image, file_name)
            
            save_path = os.path.join(OUTPUT_DIR, file_name)
            grid_img.save(save_path)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print(f"\nAll tests completed. Check results in '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()