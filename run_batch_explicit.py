import argparse
import os
import glob
import random
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Explicit Batch Sampling")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing validation images")
    parser.add_argument("--label_dir", type=str, required=True, help="Directory containing YOLO label files")
    parser.add_argument("--output_dir", type=str, default="batch_results_explicit", help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of random samples to generate")
    parser.add_argument("--control_strength", type=float, default=0.6, help="ControlNet strength")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
    
    # Filter images that have corresponding labels
    valid_pairs = []
    for img_path in image_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(args.label_dir, basename + ".txt")
        if os.path.exists(label_path):
            valid_pairs.append((img_path, label_path))
            
    print(f"Found {len(valid_pairs)} valid image-label pairs.")
    
    # Select samples
    if args.num_samples < len(valid_pairs):
        samples = random.sample(valid_pairs, args.num_samples)
    else:
        samples = valid_pairs
        
    print(f"Processing {len(samples)} samples...")
    
    for i, (img_path, label_path) in enumerate(samples):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        temp_mask_path = f"temp_mask_batch_{i}.png"
        output_path = os.path.join(args.output_dir, f"{basename}_s{args.control_strength}.png")
        
        print(f"\n--- Processing {basename} ---")
        
        # Step 1: Create Mask
        cmd_mask = [
            "python3", "create_mask.py",
            img_path,
            label_path,
            temp_mask_path
        ]
        print(f"Running: {' '.join(cmd_mask)}")
        subprocess.run(cmd_mask, check=True)
        
        # Step 2: Run Inpainting
        cmd_inpaint = [
            "python3", "inpainting_ver2/run_inpainting_controlnet.py",
            "--image", img_path,
            "--mask", temp_mask_path,
            "--output", output_path,
            "--control_strength", str(args.control_strength)
        ]
        print(f"Running: {' '.join(cmd_inpaint)}")
        subprocess.run(cmd_inpaint, check=True)
        
        # Cleanup temp mask
        if os.path.exists(temp_mask_path):
            os.remove(temp_mask_path)

    print(f"\nBatch processing complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
