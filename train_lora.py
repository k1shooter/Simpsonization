import torch
import torch.nn.functional as F
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torchvision import transforms
import config
import os
from tqdm import tqdm
import torch.nn as nn

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
    Exemplar 마스크가 없으므로, 전체 이미지 영역을 마스크(1.0)로 가정하여 학습 데이터를 준비합니다.
    """
    # 이미지 텐서
    pixel_values = transform(image).unsqueeze(0).to(device)
    
    # 잠재 공간으로 변환
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    
    # --- 핵심 수정: 마스크가 없으므로 잠재 공간 크기에 맞춰 전체 1.0 마스크 생성 ---
    _, _, h, w = latents.shape
    mask_latent = torch.ones((1, 1, h, w), device=device, dtype=latents.dtype)
    # ----------------------------------------------------------------------
    
    # Inpaint U-Net 입력 준비 (9채널)
    # 마스킹된 이미지 Latent (보존 영역 latent, 마스크 영역 0)
    # mask_latent가 1.0이므로, masked_latent_input은 latents와 동일하게 됩니다.
    masked_latent_input = latents * mask_latent
    
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

         # 9채널을 받는 새로운 conv_in 레이어 생성 (가중치는 새로 초기화됨)
        new_conv_in = nn.Conv2d(
            in_channels=9, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            bias=unet.conv_in.bias is not None # 기존 bias 설정 유지
        ).to(unet.dtype).to(config.DEVICE)

        # 새로운 conv_in으로 교체 (이 레이어의 가중치도 LoRA 학습 대상이 될 수 있습니다)
        unet.conv_in = new_conv_in
        print(f"UNet input channels successfully adapted to {unet.conv_in.in_channels}.")

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
        transforms.RandomHorizontalFlip(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
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
    
    for epoch in tqdm(range(config.TRAIN_EPOCHS)):
        # 1. Convert to Latents
        latents, mask_latent, masked_latent_input = get_train_image_and_mask(image, transform, pipeline.vae, config.DEVICE)
        
        # 2. Sample Noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=config.DEVICE)
        timesteps = timesteps.long()
        
        # 3. Add Noise
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # 4. Concatenate Noisy Latent with Masked Conditions (Inpaint Model Input)
        # U-Net input: [noisy_latents(4), masked_latent_input(4), mask_latent(1)] = (B, 9, H, W)
        unet_input = torch.cat([noisy_latents, masked_latent_input, mask_latent], dim=1)
        
        # 4. Predict Noise
        model_pred = unet(unet_input, timesteps, encoder_hidden_states).sample
        
        # 5. Loss
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            
        loss = masked_denoising_loss(model_pred.float(), target.float(), mask_latent)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{config.TRAIN_EPOCHS} - Loss: {loss.item():.4f}")
            
    # Save LoRA weights
    print(f"Saving LoRA weights to {config.OUTPUT_DIR}...")
    unet.save_pretrained(config.OUTPUT_DIR)
    print("Done.")

if __name__ == "__main__":
    train_lora()
