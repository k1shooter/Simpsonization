import torch
import torch.nn.functional as F
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import config
import os
from tqdm import tqdm
import torch.nn as nn

import glob
import numpy as np
import random

class SimpsonInpaintingDataset(Dataset):
    def __init__(self, context_root, simpson_root, transform=None):
        self.transform = transform
        
        # 배경 이미지 및 라벨 경로
        self.context_image_dir = os.path.join(context_root, "images/train") # train/images 등 실제 경로 확인 필요
        self.context_label_dir = os.path.join(context_root, "labels/train") # train/labels 등 실제 경로 확인 필요

        # 배경 이미지 목록 로드
        self.context_images = sorted(glob.glob(os.path.join(self.context_image_dir, "*.*")))
        self.context_images = [f for f in self.context_images if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 심슨 얼굴 이미지 목록 로드
        self.simpson_images = glob.glob(os.path.join(simpson_root, "**", "*.*"), recursive=True)
        self.simpson_images = [f for f in self.simpson_images if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"[Dataset] Context Images: {len(self.context_images)} at {self.context_image_dir}")
        print(f"[Dataset] Simpson Images: {len(self.simpson_images)} at {simpson_root}")

        if len(self.context_images) == 0 or len(self.simpson_images) == 0:
            raise ValueError("Data not found. Check paths in config.py")

    def __len__(self):
        return len(self.context_images)

    def load_bbox(self, img_path):
        """이미지 파일명과 매칭되는 txt 라벨 파일에서 BBox 로드"""
        basename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(basename)[0]
        # 라벨 파일 확장자는 .txt
        label_path = os.path.join(self.context_label_dir, name_no_ext + ".txt")
        
        bboxes = []
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        # YOLO: class x_center y_center width height
                        if len(parts) >= 5:
                            bboxes.append(list(map(float, parts[1:5])))
            except:
                pass
        return bboxes

    def __getitem__(self, idx):
        try:
            # 1. 배경 이미지 로드
            ctx_path = self.context_images[idx]
            context_img = Image.open(ctx_path).convert("RGB")
            w_bg, h_bg = context_img.size
            
            # 2. BBox 로드
            bboxes = self.load_bbox(ctx_path)
            
            # BBox가 없으면 해당 데이터 건너뛰기 (랜덤한 다른 데이터 반환)
            if not bboxes:
                return self.__getitem__(random.randint(0, len(self)-1))
            
            # 여러 얼굴 중 하나 랜덤 선택
            target_bbox = random.choice(bboxes)
            xc, yc, bw, bh = target_bbox
            
            # YOLO(0~1) -> 픽셀 좌표 변환
            w_box = int(bw * w_bg)
            h_box = int(bh * h_bg)
            x_center = int(xc * w_bg)
            y_center = int(yc * h_bg)
            
            # 좌상단 좌표
            x1 = x_center - w_box // 2
            y1 = y_center - h_box // 2
            
            # 좌표 보정 (이미지 벗어나지 않게)
            x1 = max(0, x1)
            y1 = max(0, y1)
            w_box = min(w_box, w_bg - x1)
            h_box = min(h_box, h_bg - y1)
            
            if w_box < 10 or h_box < 10: # 너무 작은 박스는 패스
                 return self.__getitem__(random.randint(0, len(self)-1))

            # 3. 심슨 얼굴 로드 및 리사이즈
            simp_idx = random.randint(0, len(self.simpson_images)-1)
            simpson_img = Image.open(self.simpson_images[simp_idx]).convert("RGB")
            simpson_resized = simpson_img.resize((w_box, h_box), resample=Image.LANCZOS)
            
            # 4. 합성 (Compositing) -> 이것이 모델의 정답(Target) 이미지가 됨
            composite_img = context_img.copy()
            composite_img.paste(simpson_resized, (x1, y1))
            
            # 5. 마스크 생성 (BBox 영역만 흰색)
            mask_img = Image.new("L", (w_bg, h_bg), 0)
            mask_box = Image.new("L", (w_box, h_box), 255)
            mask_img.paste(mask_box, (x1, y1))
            
            # 6. Transform (512x512 리사이즈 및 텐서 변환)
            if self.transform:
                # 이미지와 마스크는 동일하게 변형되어야 하므로 각각 적용
                composite_tensor = self.transform(composite_img)
                
                # 마스크용 transform (Interpolation=NEAREST 필수)
                mask_trans = transforms.Compose([
                    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
                    transforms.ToTensor()
                ])
                mask_tensor = mask_trans(mask_img)
            
            return composite_tensor, mask_tensor

        except Exception as e:
            print(f"Error loading {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self)-1))

# --- Masked Denoising Loss (논문 Equation 4) 함수 정의 ---
def masked_denoising_loss(noise_pred, noise_target, mask_latent):
    """논문 Equation (4) 구현: 마스킹된 영역에서만 손실 계산"""
    # mask_latent: 잠재 공간 크기의 마스크 텐서 (0 또는 1)
    
    # 마스크된 예측 노이즈와 마스크된 실제 노이즈
    masked_pred = noise_pred * mask_latent
    masked_target = noise_target * mask_latent
    
    # MSE Loss 계산 (마스크 영역에서만)
    loss = F.mse_loss(masked_pred.float(), masked_target.float(), reduction="mean")
    return loss
# --------------------------------------------------------

# --- Data Augmentation 및 Masking 유틸리티 (학습용) ---
def get_train_image_and_mask(image, transform, vae, device):
    """
    Exemplar의 일부를 랜덤하게 마스킹하여 LoRA가 인페인팅 및 블렌딩 능력을 학습하도록 합니다.
    """
    # 이미지 텐서
    pixel_values = transform(image).unsqueeze(0).to(device)
    
    # 잠재 공간으로 변환
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    
    # --- 핵심 수정: 랜덤 마스크 생성 및 올바른 masked_latent_input 계산 ---
    _, _, h, w = latents.shape
    
    # 1. 마스크 초기화 (전체 0, 즉 보존 영역)
    mask_latent = torch.zeros((1, 1, h, w), device=device, dtype=latents.dtype)
    
    # 2. 랜덤 영역을 1.0으로 설정 (마스킹 영역)
    # 마스크 크기를 잠재 공간의 1/4 ~ 1/2 사이로 랜덤 설정
    min_mask_size = h // 4
    max_mask_size = h // 2
    
    mask_size = torch.randint(min_mask_size, max_mask_size + 1, (1,)).item()
    
    # 마스크의 랜덤 시작점 계산
    if h > mask_size and w > mask_size:
        y_start = torch.randint(0, h - mask_size, (1,)).item()
        x_start = torch.randint(0, w - mask_size, (1,)).item()
        
        y_end = y_start + mask_size
        x_end = x_start + mask_size
        
        # 마스크 영역에 1.0 적용 (마스킹되어야 할 영역)
        mask_latent[:, :, y_start:y_end, x_start:x_end] = 1.0
    else:
        # 안전 장치: 마스크 사이즈가 너무 크면 전체 마스크 사용
        mask_latent = torch.ones((1, 1, h, w), device=device, dtype=latents.dtype)

    
    # 3. 마스킹된 입력 이미지 생성 (보존 영역은 exemplar, 마스킹 영역은 0)
    # mask_latent가 1.0인 곳은 0이 되고, 0인 곳은 latents가 유지됩니다.
    # 이는 UNet에게 '배경 이미지(latents)의 일부를 보존하면서 마스크 부분을 채우라'고 가르칩니다.
    masked_latent_input = latents * (1 - mask_latent) 
    # ----------------------------------------------------
    
    return latents, mask_latent, masked_latent_input
    
def train_lora():
    print(f"Loading model: {config.MODEL_ID}")
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(config.MODEL_ID)
    pipeline.to(config.DEVICE)
    
    # Freeze VAE and Text Encoder
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
    )
    
    print("Adding LoRA adapters...")
    unet = pipeline.unet
    # 입력 채널이 4개일 경우 (T2I 모델), 9개로 확장하는 새로운 conv_in을 생성
    if unet.conv_in.in_channels == 4:
        print("Adapting 4-channel UNet for 9-channel Inpainting input...")
        # 원본 conv_in 정보 추출
        out_channels = unet.conv_in.out_channels
        kernel_size = unet.conv_in.kernel_size
        padding = unet.conv_in.padding

        # 9채널을 받는 새로운 conv_in 레이어 생성
        new_conv_in = nn.Conv2d(
            in_channels=9, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            bias=unet.conv_in.bias is not None
        ).to(unet.dtype).to(config.DEVICE)

        # Initialize weights: Copy original 4 channels, zero others
        with torch.no_grad():
            new_conv_in.weight[:, :4, :, :] = unet.conv_in.weight
            new_conv_in.weight[:, 4:, :, :] = 0 # Zero init for mask/masked_image channels
            if unet.conv_in.bias is not None:
                new_conv_in.bias = unet.conv_in.bias

        unet.conv_in = new_conv_in
        print(f"UNet input channels successfully adapted to {unet.conv_in.in_channels} (with zero-init).")

    # Add conv_in to modules_to_save so it gets trained and saved
    lora_config.modules_to_save = ["conv_in"]

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    unet.train()
    
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.LEARNING_RATE)
    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    
    # Prepare Data (Single Exemplar)
    # We assume prepare_data.py has run and saved exemplar.png, or we load from dataset
    # For simplicity and robustness, let's try to load 'exemplar.png' first, if not, use dataset.
    if os.path.exists("exemplar.png"):
        image = Image.open("exemplar.png").convert("RGB")
    else:
        raise FileNotFoundError("exemplar.png not found. Please run prepare_data.py first.")
        
    # Transform
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    dataset = SimpsonInpaintingDataset(
        context_root=config.CONTEXT_MASK_ROOT,
        simpson_root=config.DATASET_ROOT,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    # Tokenize Prompt
    text_input = pipeline.tokenizer(
        config.EXEMPLAR_PROMPT,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_input.input_ids.to(config.DEVICE)
    encoder_hidden_states = pipeline.text_encoder(input_ids)[0]
    
    print(f"Starting training for {config.TRAIN_EPOCHS} epochs...")
    
    for epoch in range(config.TRAIN_EPOCHS):
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.TRAIN_EPOCHS}")
        for image, mask in progress:
            pixel_values = image.to(config.DEVICE)
            mask = mask.to(config.DEVICE)
            bsz = pixel_values.shape[0]
            
            with torch.no_grad():
                # 1. Encode image to latents
                latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipeline.vae.config.scaling_factor
            
            mask_latent = torch.nn.functional.interpolate(
                mask, size=latents.shape[-2:], mode="nearest"
            )
            mask_latent = (mask_latent > 0.5).float()  # Binarize
            masked_image_latents = latents * (1 - mask_latent)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=config.DEVICE).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 4. Concatenate Noisy Latent with Masked Conditions (Inpaint Model Input)
            # U-Net input: [noisy_latents(4), masked_latent_input(4), mask_latent(1)] = (B, 9, H, W)
            unet_input = torch.cat([noisy_latents, masked_image_latents, mask_latent], dim=1)
            
            # 4. Predict Noise
            batch_encoder_hidden_states = encoder_hidden_states.repeat(bsz, 1, 1)
            model_pred = unet(unet_input, timesteps, encoder_hidden_states).sample
            
            loss = masked_denoising_loss(model_pred, noise, mask_latent)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            progress.set_postfix({"loss": loss.item()})
    # Save LoRA weights
    print(f"Saving LoRA weights to {config.OUTPUT_DIR}...")
    unet.save_pretrained(config.OUTPUT_DIR)
    print("Done.")

if __name__ == "__main__":
    train_lora()
