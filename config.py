import torch
import os

# Model Settings
MODEL_ID = "Manojb/stable-diffusion-2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LoRA Settings
LORA_RANK = 16
LORA_ALPHA = 16
TARGET_MODULES = ["to_k", "to_v"] # Targeting attention layers
LORA_DROPOUT = 0.0

# Training Settings
TRAIN_EPOCHS = 10 # Enough to learn the style/identity of a single image
LEARNING_RATE = 5e-5
OUTPUT_DIR = "lora_weights"
LORA_WEIGHTS_NAME = "simpson_lora.pt"

# Data Settings
DATASET_ROOT = "./data/datasets/kostastokis/simpsons-faces/versions/1/cropped"
CONTEXT_MASK_ROOT = "/root/vis_contest/data/datasets/fareselmenshawii/face-detection-dataset/versions/3"
EXEMPLAR_PROMPT = "A photo of a simpson face" # Fixed prompt as requested
NEGATIVE_PROMPT = "blurry image, disfigured face, bad anatomy, low resolution, deformed body features, poorly drawn face, bad composition"

# Inpainting Settings
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 1
