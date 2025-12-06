# 파일: run_inpainting_repaint.py
import argparse
from inpainting_pipeline_repaint import CustomRepaintPipeline
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--mask", type=str, required=True)
    parser.add_argument("--output", type=str, default="output_repaint.png")
    parser.add_argument("--lora", type=str, default=config.OUTPUT_DIR)
    parser.add_argument("--prompt", type=str, default=config.EXEMPLAR_PROMPT)
    parser.add_argument("--strength", type=float, default=1.0) # RePaint는 보통 전체 스텝을 돕니다
    args = parser.parse_args()
    
    # 모델 ID는 config에서 가져오되, SD2 Base인지 확인
    print(f"Using Model: {config.MODEL_ID}")
    
    pipeline = CustomRepaintPipeline(config.MODEL_ID, args.lora, device=config.DEVICE)
    
    image = Image.open(args.image).convert("RGB")
    
    result = pipeline.inpaint(
        image=image,
        mask_path=args.mask,
        prompt=args.prompt,
        num_inference_steps=config.NUM_INFERENCE_STEPS,
        guidance_scale=config.GUIDANCE_SCALE,
        strength=args.strength
    )
    
    result.save(args.output)
    print(f"Saved result to {args.output}")

if __name__ == "__main__":
    main()
