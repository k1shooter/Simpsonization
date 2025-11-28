import argparse
from inpainting_pipeline import CustomInpaintingPipeline
from PIL import Image
import config
import os

def main():
    parser = argparse.ArgumentParser(description="Simpson Inpainting")
    parser.add_argument("--image", type=str, default="puppy.png", help="Path to background image")
    parser.add_argument("--mask", type=str, default="mask.png", help="Path to mask image")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save output")
    parser.add_argument("--lora", type=str, default=config.OUTPUT_DIR, help="Path to LoRA weights")
    parser.add_argument("--prompt", type=str, default=config.EXEMPLAR_PROMPT, help="Prompt for inpainting")
    parser.add_argument("--exemplar", type=str, default="exemplar.png", help="Path to exemplar image for PNI")
    args = parser.parse_args()
    
    print(f"Initializing Pipeline with model {config.MODEL_ID}...")
    # Check if LoRA exists
    if not os.path.exists(args.lora):
        print(f"Warning: LoRA weights not found at {args.lora}. Using base model only.")
        lora_path = None
    else:
        lora_path = args.lora
        
    pipeline = CustomInpaintingPipeline(config.MODEL_ID, lora_path, device=config.DEVICE)
    
    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert("RGB")
    mask = args.mask
    print(f"Loading mask: {args.mask}")
    
    exemplar = None
    if os.path.exists(args.exemplar):
        print(f"Loading exemplar for PNI: {args.exemplar}")
        exemplar = Image.open(args.exemplar).convert("RGB")
    else:
        print(f"Warning: Exemplar not found at {args.exemplar}. PNI will be disabled.")
    
    print(f"Inpainting with prompt: '{args.prompt}'...")
    result = pipeline.inpaint(image, mask, args.prompt, exemplar_image=exemplar, num_inference_steps=config.NUM_INFERENCE_STEPS, guidance_scale=config.GUIDANCE_SCALE)
    
    result.save(args.output)
    print(f"Saved result to {args.output}")

if __name__ == "__main__":
    main()
