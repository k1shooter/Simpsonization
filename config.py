import torch
import os

# Model Settings
MODEL_ID = "Manojb/stable-diffusion-2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LoRA Settings
LORA_RANK = 16
LORA_ALPHA = 16
TARGET_MODULES = ["to_k", "to_v", "to_q", "to_out.0"] # Targeting attention layers
LORA_DROPOUT = 0.0

# Training Settings
TRAIN_EPOCHS = 200 # Enough to learn the style/identity of a single image
LEARNING_RATE = 1e-4
OUTPUT_DIR = "lora_weights"
LORA_WEIGHTS_NAME = "simpson_lora.pt"

# Data Settings
DATASET_ROOT = "./data/datasets/kostastokis/simpsons-faces/versions/1/cropped"
EXEMPLAR_INDEX = 0 # Fix one index for the exemplar
EXEMPLAR_PROMPT = "A photo of a simpson face" # Fixed prompt as requested

# Inpainting Settings
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
