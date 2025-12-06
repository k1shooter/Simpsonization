# 파일: config.py
import torch
import os

# --- Model Settings ---
MODEL_ID = "Manojb/stable-diffusion-2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- LoRA Settings (논문 3.3.1절 준수) ---
# --- LoRA Settings (논문 3.3.1절 준수) ---
LORA_RANK = 16
LORA_ALPHA = 8
# [FIX] 논문대로 Key, Value만 학습 -> Base Model의 3D 표현력 보존
TARGET_MODULES = ["to_k", "to_v"] 
LORA_DROPOUT = 0.0

# --- Training Settings ---
# 데이터셋이 작다면 에포크를 충분히(100~300) 주세요
TRAIN_EPOCHS = 3 
LEARNING_RATE = 1e-4
OUTPUT_DIR = "lora_weights"

# --- Data Settings ---
# --- Data Settings ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIMPSON_FACE_ROOT = os.path.join(BASE_DIR, "data/datasets/kostastokis/simpsons-faces/versions/1/cropped")
SIMPSON_MASK_ROOT = os.path.join(BASE_DIR, "data/datasets/kostastokis/simpsons-faces/versions/1/simplified")
CONTEXT_MASK_ROOT = os.path.join(BASE_DIR, "data/datasets/fareselmenshawii/face-detection-dataset/versions/3")

# [핵심] 3D 입체감을 강제하는 프롬프트
EXEMPLAR_PROMPT = "hyperrealistic 3d render of a simpson face, claymation style, c4d, blender, volumetric lighting, high fidelity, 8k, detailed texture, 3d modeling, unreal engine, pixar style, disney style" 

# --- Inference Settings ---
NUM_INFERENCE_STEPS = 200
# [핵심] 높은 Guidance Scale로 3D 프롬프트를 강력하게 반영
GUIDANCE_SCALE = 16.0
NEGATIVE_PROMPT = "2d, flat, drawing, sketch, paper, cartoon, anime, low quality, distorted, blurry, bad anatomy"