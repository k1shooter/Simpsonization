# 파일: train_lora_plus.py
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import config_v2 as config
import os
import glob
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import random

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
                # 1. Resize (Common)
                resize = transforms.Resize((512, 512))
                image = resize(image)
                mask = resize(mask)
                
                # 2. Random Horizontal Flip (Synchronized)
                if random.random() > 0.5:
                    image = transforms.functional.hflip(image)
                    mask = transforms.functional.hflip(mask)
                
                # 3. Color Jitter (Image Only)
                color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
                image = color_jitter(image)
                
                # 4. ToTensor & Normalize
                to_tensor = transforms.ToTensor()
                normalize = transforms.Normalize([0.5], [0.5])
                
                image_tensor = normalize(to_tensor(image))
                
                # Mask Transform (Just ToTensor, already resized)
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

def train_lora_plus():
    print(f"Loading model: {config.MODEL_ID}")
    # T2I 파이프라인 (스타일/개념 학습용)
    pipeline = StableDiffusionPipeline.from_pretrained(config.MODEL_ID)
    pipeline.to(config.DEVICE)
    
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    
    # LoRA Config (논문 준수)
    lora_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.TARGET_MODULES, # ["to_k", "to_v"]
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        modules_to_save=["conv_in"] # Inpainting 호환성 위해 저장
    )
    
    # UNet 9채널 적응
    unet = pipeline.unet
    if unet.conv_in.in_channels == 4:
        print("Adapting UNet 4ch -> 9ch...")
        with torch.no_grad():
            old_conv = unet.conv_in
            new_conv = nn.Conv2d(
                in_channels=9, 
                out_channels=old_conv.out_channels, 
                kernel_size=old_conv.kernel_size, 
                padding=old_conv.padding, 
                bias=old_conv.bias is not None
            ).to(config.DEVICE)
            
            new_conv.weight[:, :4, :, :] = old_conv.weight
            new_conv.weight[:, 4:, :, :] = 0 
            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias
            
            unet.conv_in = new_conv

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    unet.train()
    
    # --- LoRA+ Optimizer Setup ---
    print("Initializing LoRA+ Optimizer...")
    lora_plus_lr_ratio = 16.0
    
    # Group parameters
    # Group 1: LoRA B matrices (High LR)
    # Group 2: Everything else (Base LR) - includes LoRA A, conv_in, etc.
    
    params_lora_b = []
    params_rest = []
    
    for name, param in unet.named_parameters():
        if not param.requires_grad:
            continue
            
        if "lora_B" in name:
            params_lora_b.append(param)
        else:
            params_rest.append(param)
            
    optimizer_grouped_parameters = [
        {
            "params": params_lora_b,
            "lr": config.LEARNING_RATE * lora_plus_lr_ratio,
            "name": "lora_B_group"
        },
        {
            "params": params_rest,
            "lr": config.LEARNING_RATE,
            "name": "rest_group"
        },
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    print(f"LoRA+ Configured: Matrix B LR = {config.LEARNING_RATE * lora_plus_lr_ratio}, Others LR = {config.LEARNING_RATE}")
    # -----------------------------
    
    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    
    # Dataset
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
            
            # Masked Image
            masked_image_latents = latents * (1 - mask_latent)
            
            # Noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=config.DEVICE).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # UNet Input
            unet_input = torch.cat([noisy_latents, masked_image_latents, mask_latent], dim=1)
            
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
            
    output_dir = "lora_weights_plus"
    unet.save_pretrained(output_dir)
    print(f"Saved LoRA+ weights to {output_dir}")
    print("Done.")

if __name__ == "__main__":
    train_lora_plus()
