import sys
from PIL import Image, ImageDraw
import os

def create_mask(image_path, label_path, output_path):
    # Load image to get dimensions
    try:
        img = Image.open(image_path)
        width, height = img.size
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Create empty black mask
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Read label file
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO format: class x_center y_center w h
                # We ignore class
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                
                # Convert to pixel coordinates
                x1 = int((x_center - w/2) * width)
                y1 = int((y_center - h/2) * height)
                x2 = int((x_center + w/2) * width)
                y2 = int((y_center + h/2) * height)
                
                # Draw white rectangle
                draw.rectangle([x1, y1, x2, y2], fill=255)
                
        mask.save(output_path)
        print(f"Mask saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing label: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python create_mask.py <image_path> <label_path> <output_path>")
    else:
        create_mask(sys.argv[1], sys.argv[2], sys.argv[3])
