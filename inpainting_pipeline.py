# inpainting_pipeline.py 파일의 191번째 줄 부근을 수정합니다.
# 수정 전: image = self.pipe.vae.decode(latents.float()).sample 
# 수정 후: image = self.pipe.vae.decode(latents).sample 
# -------------------------------------------------------------

import torch
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler, UNet2DConditionModel
import numpy as np
from PIL import Image
import config
from tqdm import tqdm
from peft import PeftModel 
import torch.nn as nn 
import os
# --- Utility: YOLO BBox to Binary Mask ---
def yolo_to_mask_image(txt_path, image_size=(512, 512)):
    """
    YOLO 형식 주석 (class x_c y_c w h)을 읽어 하나의 이진 PIL 마스크 이미지로 변환합니다.
    """
    img_w, img_h = image_size
    mask_arr = np.zeros(image_size, dtype=np.uint8)
    
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Warning: YOLO label file not found at {txt_path}. Using all-black mask.")
        return Image.fromarray(mask_arr, 'L') 
        
    for line in lines:
        try:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            # Assuming format: class x_c y_c w h
            _, x_c, y_c, w, h = map(float, parts[:5])
            
            x_min = int((x_c - w/2) * img_w)
            y_min = int((y_c - h/2) * img_h)
            x_max = int((x_c + w/2) * img_w)
            y_max = int((y_c + h/2) * img_h)
            
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_w, x_max)
            y_max = min(img_h, y_max)
            
            # Fill the bounding box region with 255 (white for mask)
            mask_arr[y_min:y_max, x_min:x_max] = 255
            
        except ValueError as e:
            continue
            
    return Image.fromarray(mask_arr, 'L')
# -----------------------------------------------

class CustomInpaintingPipeline:
    def __init__(self, base_model_id, lora_weights_path, device="cuda"):
        self.device = device
        print(f"Loading Inpainting pipeline from {base_model_id}...")
        
        # StableDiffusionPipeline 로드 (Base 모델이므로)
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
        self.pipe.to(device)
        
        if self.pipe.unet.config.in_channels == 4:
            print("Detected 4-channel model. Re-instantiating as 9-channel Inpainting UNet...")
            
            # 1. 기존 설정 가져오기 및 수정
            new_config = dict(self.pipe.unet.config)
            new_config["in_channels"] = 9
            
            # 2. 새로운 9채널 UNet 생성 (랜덤 가중치 상태)
            new_unet = UNet2DConditionModel.from_config(new_config).to(device, dtype=self.pipe.unet.dtype)
            
            # 3. 가중치 이식 (Weight Transfer)
            # 기존 4채널 가중치를 가져옴
            state_dict = self.pipe.unet.state_dict()
            
            # conv_in 레이어의 가중치 형상을 9채널로 확장
            old_conv_in = state_dict["conv_in.weight"] # [320, 4, 3, 3]
            new_conv_in = torch.zeros((320, 9, 3, 3), device=device, dtype=old_conv_in.dtype)
            
            # 기존 4채널 복사, 나머지 5채널은 0으로 초기화
            new_conv_in[:, :4, :, :] = old_conv_in
            new_conv_in[:, 4:, :, :] = 0 
            
            # 수정된 가중치를 state_dict에 업데이트
            state_dict["conv_in.weight"] = new_conv_in
            
            # 새로운 모델에 가중치 로드
            new_unet.load_state_dict(state_dict)
            
            # 파이프라인의 UNet 교체
            self.pipe.unet = new_unet
            print("UNet successfully adapted to 9 channels.")
        # ---------------------------------------------------

        # 2. LoRA 로드
        if lora_weights_path and os.path.exists(lora_weights_path):
            print(f"Loading LoRA from: {lora_weights_path}")
            self.pipe.unet = PeftModel.from_pretrained(self.pipe.unet, lora_weights_path)
            self.pipe.unet = self.pipe.unet.merge_and_unload()
            print("LoRA merged successfully.")
        else:
            print(f"Warning: LoRA weights not found at {lora_weights_path}")
        
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
    
    @torch.no_grad()
    def inpaint(self, image, mask_path, prompt, num_inference_steps=50, guidance_scale=7.5):
        # 1. 마스크 처리
        image_size = (512, 512)
        if mask_path.endswith('.txt'):
            mask_image = yolo_to_mask_image(mask_path, image_size)
        else:
            mask_image = Image.open(mask_path).convert("L").resize(image_size, Image.NEAREST)
            
        # 2. 이미지 리사이즈
        input_image = image.resize(image_size, Image.LANCZOS)
        
        # 3. 파이프라인 실행 (표준 인페인팅)
        # PNI 등 별도 노이즈 주입 없이, 학습된 모델이 알아서 채우도록 함
        result = self.pipe(
            prompt=prompt,
            image=input_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=config.NEGATIVE_PROMPT
        ).images[0]
        
        return result
        
    # @torch.no_grad()
    # def inpaint(self, image, mask_path, prompt, num_inference_steps=50, guidance_scale=7.5, seed=None):
        
    #     image_size_for_mask = (512, 512)

    #     if mask_path.endswith('.txt'):
    #         mask_image = yolo_to_mask_image(mask_path, image_size)
    #     else:
    #         mask_image = Image.open(mask_path).convert("L").resize(image_size, Image.NEAREST)
        
    #     # 2. Preprocess: Resize images/mask to standard 512x512
    #     input_image = image.resize((512, 512), resample=Image.LANCZOS)
        
    #     # VAE Encode Background and Mask
    #     img_tensor = torch.from_numpy(np.array(input_image)).float() / 127.5 - 1.0
    #     img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.pipe.vae.dtype)
        
    #     mask_tensor = torch.from_numpy(np.array(input_mask.convert("L"))).float() / 255.0
    #     mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device, dtype=self.pipe.unet.dtype)
        
    #     with torch.no_grad():
    #         latents_bg = self.pipe.vae.encode(img_tensor).latent_dist.sample()
    #         latents_bg = latents_bg * self.pipe.vae.config.scaling_factor
        
    #     mask_latent = torch.nn.functional.interpolate(mask_tensor, size=latents_bg.shape[-2:], mode="nearest")
        
    #     # 3. Initial Noise (PNI or Random)
    #     initial_latents = None
        
    #     self.pipe.scheduler.set_timesteps(num_inference_steps)
    #     timesteps = self.pipe.scheduler.timesteps
    #     t_start = timesteps[0] 

    #     if exemplar_image is not None:
    #         print("Applying Prior Noise Initialization (PNI)...")
            
    #         mask_arr = np.array(input_mask.convert("L")) 
    #         rows = np.any(mask_arr, axis=1)
    #         cols = np.any(mask_arr, axis=0)
            
    #         if np.any(rows) and np.any(cols):
    #             ymin, ymax = np.where(rows)[0][[0, -1]]
    #             xmin, xmax = np.where(cols)[0][[0, -1]]
    #             h, w = ymax - ymin, xmax - xmin
                
    #             exemplar_resized = exemplar_image.resize((w, h), resample=Image.LANCZOS)
    #             composite = input_image.copy()
    #             composite.paste(exemplar_resized, (xmin, ymin))
                
    #             comp_tensor = torch.from_numpy(np.array(composite)).float() / 127.5 - 1.0
    #             comp_tensor = comp_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.pipe.vae.dtype)
                
    #             with torch.no_grad():
    #                 latents_comp = self.pipe.vae.encode(comp_tensor).latent_dist.sample()
    #                 latents_comp = latents_comp * self.pipe.vae.config.scaling_factor
                
    #             noise = torch.randn(latents_comp.shape, device=self.device, dtype=latents_comp.dtype, generator=None)
    #             initial_latents = self.pipe.scheduler.add_noise(latents_comp, noise, t_start) 
    #         else:
    #             print("Warning: Mask is empty. Skipping PNI and using random noise.")
        
    #     if initial_latents is None:
    #         initial_latents = torch.randn(latents_bg.shape, device=self.device, dtype=latents_bg.dtype, generator=None)
            
    #     # 4. Text Embeddings
    #     text_input = self.pipe.tokenizer(
    #         prompt,
    #         padding="max_length",
    #         max_length=self.pipe.tokenizer.model_max_length,
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    #     text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
        
    #     uncond_input = self.pipe.tokenizer(
    #         [config.NEGATIVE_PROMPT], padding="max_length", max_length=text_input.input_ids.shape[-1], return_tensors="pt"
    #     )
    #     uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
    #     text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
    #     # 5. Prepare Inpainting Condition (Masked Image Latent + Mask)
    #     masked_image_latent = latents_bg * (1 - mask_latent) 
    #     latents = initial_latents

    #     self.pipe.scheduler.set_timesteps(num_inference_steps)
    #     timesteps = self.pipe.scheduler.timesteps
        
    #     # 6. Denoising Loop (PNI Latent 사용 및 Inpainting Blending)
    #     print("Inpainting...")
    #     for t in tqdm(timesteps):
    #         latent_model_input = torch.cat([latents] * 2)
    #         latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            
    #         # UNet Input: [noisy_latent_4, masked_image_latent_4, mask_1] = (B, 9, H, W)
    #         unet_input = torch.cat([latent_model_input, masked_image_latent.repeat(2, 1, 1, 1), mask_latent.repeat(2, 1, 1, 1)], dim=1)
            
    #         noise_pred = self.pipe.unet(unet_input, t, encoder_hidden_states=text_embeddings).sample
            
    #         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #         noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
    #         latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
    #         latents = latents * mask_latent + latents_bg * (1 - mask_latent)

    #     # latents = latents * mask_latent + latents_bg * (1 - mask_latent)
    #     # 7. Decode
    #     latents = 1 / self.pipe.vae.config.scaling_factor * latents
    #     # FIX: .float() 제거
    #     image = self.pipe.vae.decode(latents).sample 
        
    #     image = (image / 2 + 0.5).clamp(0, 1)
    #     image = image.cpu().permute(0, 2, 3, 1).numpy()
    #     image = (image * 255).round().astype("uint8")
        
    #     return Image.fromarray(image[0])