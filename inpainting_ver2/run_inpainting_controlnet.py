# 파일: run_inpainting_controlnet.py
import argparse
import torch
from inpainting_controlnet import CustomControlNetPipeline
import config_v2 as config
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Simpson Face Inpainting with ControlNet")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--mask", type=str, required=True, help="Path to mask file (txt or image)")
    parser.add_argument("--output", type=str, default="output_controlnet.png", help="Path to save output")
    parser.add_argument("--prompt", type=str, default=config.EXEMPLAR_PROMPT, help="Prompt for generation")
    parser.add_argument("--strength", type=float, default=0.75, help="Denoising strength")
    parser.add_argument("--control_strength", type=float, default=0.6, help="ControlNet strength")
    parser.add_argument("--guidance_scale", type=float, default=16.0, help="Guidance scale")
    
    args = parser.parse_args()
    
    device = config.DEVICE
    
    # Initialize Pipeline
    pipeline = CustomControlNetPipeline(
        base_model_id=config.MODEL_ID,
        lora_weights_path=config.OUTPUT_DIR,
        device=device
    )
    
    # Load Image
    image = Image.open(args.image).convert("RGB")
    
    # Run Inference
    result = pipeline.inpaint(
        image=image,
        mask_path=args.mask,
        prompt=args.prompt,
        strength=args.strength,
        control_strength=args.control_strength,
        guidance_scale=args.guidance_scale
    )
    
    # Save
    result.save(args.output)
    print(f"Saved result to {args.output}")

if __name__ == "__main__":
    main()
