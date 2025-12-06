# 파일: test_repaint_batch.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import glob
import argparse
from inpainting_pipeline_repaint import CustomRepaintPipeline
from PIL import Image
import config
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to test")
    parser.add_argument("--output_dir", type=str, default="test_results_repaint", help="Directory to save results")
    parser.add_argument("--lora", type=str, default=config.OUTPUT_DIR, help="Path to LoRA weights")
    args = parser.parse_args()
    
    # 데이터셋 경로 (Validation Set)
    val_img_dir = "/root/vis_contest/data/datasets/fareselmenshawii/face-detection-dataset/versions/3/images/val"
    val_lbl_dir = "/root/vis_contest/data/datasets/fareselmenshawii/face-detection-dataset/versions/3/labels/val"
    
    # 이미지 파일 리스트
    img_files = glob.glob(os.path.join(val_img_dir, "*.jpg"))
    if not img_files:
        print("No images found in validation directory.")
        return

    # 랜덤 선택
    selected_files = random.sample(img_files, min(len(img_files), args.num_images))
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 파이프라인 로드
    print(f"Loading Pipeline with LoRA: {args.lora}")
    pipeline = CustomRepaintPipeline(config.MODEL_ID, args.lora, device=config.DEVICE)
    
    print(f"Processing {len(selected_files)} images...")
    
    for img_path in tqdm(selected_files):
        basename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(basename)[0]
        
        # 라벨 파일 찾기
        lbl_path = os.path.join(val_lbl_dir, name_no_ext + ".txt")
        if not os.path.exists(lbl_path):
            print(f"Label not found for {basename}, skipping.")
            continue
            
        # 이미지 로드
        image = Image.open(img_path).convert("RGB")
        
        # Inpainting 수행
        try:
            result = pipeline.inpaint(
                image=image,
                mask_path=lbl_path, # .txt 파일 경로 전달 -> 내부에서 타원 마스크 생성
                prompt=config.EXEMPLAR_PROMPT,
                num_inference_steps=config.NUM_INFERENCE_STEPS,
                guidance_scale=config.GUIDANCE_SCALE,
                strength=1.0
            )
            
            # 결과 저장
            save_path = os.path.join(args.output_dir, f"result_{basename}")
            result.save(save_path)
            
        except Exception as e:
            print(f"Error processing {basename}: {e}")

    print(f"All done! Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()
