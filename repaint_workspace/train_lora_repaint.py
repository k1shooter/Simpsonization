# 파일: train_lora_repaint.py
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import itertools
import math
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import random
import glob
from tqdm import tqdm
import numpy as np
import torch.nn as nn

class SimpsonFaceDataset(Dataset):
    def __init__(self, face_root, mask_root, transform=None):
        self.face_root = face_root
        self.mask_root = mask_root
        self.transform = transform
        
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(glob.glob(os.path.join(face_root, "**", ext), recursive=True))
        self.image_paths = sorted(list(set(self.image_paths)))
        
        print(f"Found {len(self.image_paths)} Simpson images.")

    def __len__(self):
        return len(self.image_paths)

    def process_simplified_mask(self, mask_img):
        """Simplified PNG(노랑/흰색) -> Binary Mask 변환"""
        mask_arr = np.array(mask_img.convert("RGB"))
        # R>150 & G>150 이면 얼굴(노랑) 또는 눈/이(흰색)
        face_condition = (mask_arr[:, :, 0] > 150) & (mask_arr[:, :, 1] > 150)
        
        binary_mask = np.zeros_like(mask_arr[:, :, 0])
        binary_mask[face_condition] = 255
        return Image.fromarray(binary_mask).convert("L")

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            file_name = os.path.basename(img_path)
            
            # 1. 이미지 로드
            image = Image.open(img_path).convert("RGB")
            
            # 2. 마스크 로드
            base_name = os.path.splitext(file_name)[0]
            mask_path = None
            
            # simplified 폴더 검색
            potential_masks = glob.glob(os.path.join(self.mask_root, "**", base_name + ".*"), recursive=True)
            if potential_masks:
                mask_path = potential_masks[0]
            
            if mask_path and os.path.exists(mask_path):
                raw_mask = Image.open(mask_path)
                raw_mask = raw_mask.resize(image.size, Image.NEAREST)
                mask = self.process_simplified_mask(raw_mask)
            else:
                # 마스크 없으면 전체 사용
                mask = Image.new("L", image.size, 255)

            # 3. Transform
            if self.transform:
                # [FIX] 이미지와 마스크 동기화된 Transform 적용
                
                # 1. Resize (공통)
                resize = transforms.Resize((512, 512))
                image = resize(image)
                mask = resize(mask) # 마스크도 PIL 상태에서 Resize
                
                # 2. Random Horizontal Flip (동기화)
                if random.random() > 0.5:
                    image = transforms.functional.hflip(image)
                    mask = transforms.functional.hflip(mask)
                
                # 3. Color Jitter (이미지만)
                color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
                image = color_jitter(image)
                
                # 4. ToTensor & Normalize
                to_tensor = transforms.ToTensor()
                normalize = transforms.Normalize([0.5], [0.5])
                
                image_tensor = normalize(to_tensor(image))
                mask_tensor = to_tensor(mask)
            
            return image_tensor, mask_tensor

        except Exception as e:
            return self.__getitem__(0)

def masked_denoising_loss(noise_pred, noise_target, mask_latent):
    # 논문 Eq 4: 마스크 영역(얼굴)만 학습
    masked_pred = noise_pred * mask_latent
    masked_target = noise_target * mask_latent
    loss = F.mse_loss(masked_pred.float(), masked_target.float(), reduction="mean")
    return loss

def train_lora_repaint():
    print(f"Loading model for Repaint Training (4ch): {config.MODEL_ID}")
    
    # [변경] 기본 SD 파이프라인 사용 (4채널)
    pipeline = StableDiffusionPipeline.from_pretrained(config.MODEL_ID)
    pipeline.to(config.DEVICE)
    
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    
    # LoRA Config
    lora_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.TARGET_MODULES, # ["to_k", "to_v"]
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        # [변경] modules_to_save 제거 (conv_in 학습 불필요)
    )
    
    # [변경] UNet 9채널 확장 로직 제거 -> 순정 4채널 사용
    unet = pipeline.unet
    
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    unet.train()
    
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.LEARNING_RATE)
    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    
    # Dataset
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    dataset = SimpsonFaceDataset(
        face_root=config.SIMPSON_FACE_ROOT,
        mask_root=config.SIMPSON_MASK_ROOT,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    
    # Text Embeddings
    text_input = pipeline.tokenizer(
        config.EXEMPLAR_PROMPT,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = text_input.input_ids.to(config.DEVICE)
    encoder_hidden_states = pipeline.text_encoder(input_ids)[0]
    
    print(f"Starting training for {config.TRAIN_EPOCHS} epochs...")
    
    for epoch in range(config.TRAIN_EPOCHS):
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for image, mask in progress:
            pixel_values = image.to(config.DEVICE)
            mask = mask.to(config.DEVICE)
            bsz = pixel_values.shape[0]
            
            # Latents
            with torch.no_grad():
                latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipeline.vae.config.scaling_factor
            
            # Mask Latent
            mask_latent = torch.nn.functional.interpolate(mask, size=latents.shape[-2:], mode="nearest")
            mask_latent = (mask_latent > 0.5).float()
            
            # Noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=config.DEVICE).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # [변경] UNet Input: 4채널 (noisy_latents only)
            # Inpainting 모델이 아니므로 masked_image_latents, mask_latent를 concat하지 않음
            unet_input = noisy_latents
            
            # Predict
            batch_encoder_hidden_states = encoder_hidden_states.repeat(bsz, 1, 1)
            model_pred = unet(unet_input, timesteps, batch_encoder_hidden_states).sample
            
            # Loss
            target = noise
            loss = masked_denoising_loss(model_pred, target, mask_latent)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            progress.set_postfix({"loss": loss.item()})
            
    unet.save_pretrained(config.OUTPUT_DIR)
    print("Done.")

if __name__ == "__main__":
    train_lora_repaint()
