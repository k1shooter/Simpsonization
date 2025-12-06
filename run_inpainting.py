# 파일: run_inpainting.py
import argparse
from inpainting_pipeline import CustomInpaintingPipeline
from PIL import Image
import config
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--mask", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--lora", type=str, default=config.OUTPUT_DIR)
    parser.add_argument("--prompt", type=str, default=config.EXEMPLAR_PROMPT)
    parser.add_argument("--strength", type=float, default=0.75) # PNI 강도
    args = parser.parse_args()
    
    pipeline = CustomInpaintingPipeline(config.MODEL_ID, args.lora, device=config.DEVICE)
    
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