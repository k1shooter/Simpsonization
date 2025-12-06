import argparse
import torch
import sys
sys.path.append("inpainting_ver2")
from inpainting_controlnet import CustomControlNetPipeline
import config_v2 as config
from PIL import Image
import os

def main():
    parser = argparse.ArgumentParser(description="Style Improvement Test")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--mask", type=str, required=True, help="Path to mask file")
    args = parser.parse_args()
    
    device = config.DEVICE
    
    # Initialize Pipeline
    pipeline = CustomControlNetPipeline(
        base_model_id=config.MODEL_ID,
        lora_weights_path=config.OUTPUT_DIR,
        device=device
    )
    
    image = Image.open(args.image).convert("RGB")
    
    # Test Settings
    control_strengths = [0.4, 0.6, 0.8, 1.0]
    
    # Enhanced Prompt
    base_prompt = "hyperrealistic 3d render of a simpson character face, big circular eyes, yellow skin, overbite, claymation style, c4d, blender, volumetric lighting, pixar style"
    
    for strength in control_strengths:
        print(f"Testing ControlNet Strength: {strength}")
        
        result = pipeline.inpaint(
            image=image,
            mask_path=args.mask,
            prompt=base_prompt,
            strength=0.8, # Slightly higher denoising to allow more change
            control_strength=strength,
            guidance_scale=16.0
        )
        
        output_filename = f"style_test_s{strength}.png"
        result.save(output_filename)
        print(f"Saved {output_filename}")

if __name__ == "__main__":
    main()
