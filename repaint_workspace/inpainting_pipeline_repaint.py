# 파일: inpainting_pipeline_repaint.py
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image, ImageFilter, ImageDraw
import config
from peft import PeftModel 
from tqdm import tqdm   

def yolo_to_mask_image(txt_path, image_size=(512, 512)):
    img_w, img_h = image_size
    mask_arr = np.zeros(image_size, dtype=np.uint8)
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            _, x_c, y_c, w, h = map(float, parts[:5])
            x_min = int((x_c - w/2) * img_w)
            y_min = int((y_c - h/2) * img_h)
            x_max = int((x_c + w/2) * img_w)
            y_max = int((y_c + h/2) * img_h)
            
            # [변경] 사각형 대신 둥근 사각형(Rounded Rectangle) 마스크 그리기
            # 빈 이미지에 둥근 사각형을 그리고 기존 마스크에 합침
            temp_mask = Image.new('L', image_size, 0)
            draw = ImageDraw.Draw(temp_mask)
            
            # Box 확대 (1.2배 - 사용자 변경 사항 유지)
            box_w = x_max - x_min
            box_h = y_max - y_min
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
            new_w = box_w * 1.2
            new_h = box_h * 1.2
            
            nx_min = max(0, cx - new_w / 2)
            ny_min = max(0, cy - new_h / 2)
            nx_max = min(img_w, cx + new_w / 2)
            ny_max = min(img_h, cy + new_h / 2)
            
            # 둥근 모서리 반경 (짧은 변의 20%)
            radius = min(new_w, new_h) * 0.2
            
            draw.rounded_rectangle((nx_min, ny_min, nx_max, ny_max), radius=radius, fill=255)
            
            # [NEW] 마스크 자체를 부드럽게 처리 (Soft Mask for Input)
            temp_mask = temp_mask.filter(ImageFilter.GaussianBlur(radius=15)) # 강한 블러링으로 경계 완화
            
            # Numpy 변환 후 합치기 (여러 얼굴일 경우)
            temp_arr = np.array(temp_mask)
            mask_arr = np.maximum(mask_arr, temp_arr)
    except Exception:
        pass
    return Image.fromarray(mask_arr, 'L')

class CustomRepaintPipeline:
    def __init__(self, base_model_id, lora_weights_path, device="cuda"):
        self.device = device
        print(f"Loading Repaint Pipeline (SD2 Base): {base_model_id}")
        
        # [변경] InpaintPipeline 대신 기본 Pipeline 사용 (4채널 모델 그대로 사용)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model_id, 
            torch_dtype=torch.float16,
            safety_checker=None
        )
        self.pipe.to(device)
        
        # LoRA 로드
        if lora_weights_path and os.path.exists(lora_weights_path):
            try:
                self.pipe.unet = PeftModel.from_pretrained(self.pipe.unet, lora_weights_path)
                self.pipe.unet = self.pipe.unet.merge_and_unload()
                print("LoRA merged successfully.")
            except Exception as e:
                print(f"Error loading LoRA: {e}")
        
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
    
    @torch.no_grad()
    def inpaint(self, image, mask_path, prompt, num_inference_steps=50, guidance_scale=15.0, strength=0.75):
        orig_w, orig_h = image.size
        target_size = (512, 512)
        
        # Load Mask
        if mask_path.endswith('.txt'):
            mask_image = yolo_to_mask_image(mask_path, target_size)
        else:
            mask_image = Image.open(mask_path).convert("L").resize(target_size, Image.NEAREST)
            
        input_image = image.resize(target_size, Image.LANCZOS)
        
        # Prepare Tensors
        # 1. Image -> Latent
        image_np = np.array(input_image.convert("RGB")).astype(np.float32) / 127.5 - 1.0
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).unsqueeze(0).to(self.device, dtype=self.pipe.unet.dtype)
        
        init_latents = self.pipe.vae.encode(image_tensor).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        
        # 2. Mask -> Latent Size
        mask_np = np.array(mask_image.convert("L")).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device, dtype=self.pipe.unet.dtype)
        
        # Resize mask to latent shape
        # [핵심 변경] Nearest -> Bilinear & Threshold 제거 (Soft Blending)
        mask_latent = torch.nn.functional.interpolate(mask_tensor, size=init_latents.shape[-2:], mode="bilinear", align_corners=False)
        
        # [FIX] Loop 내에서는 Hard Mask 사용 (배경 노이즈 침범 방지)
        mask_latent_hard = (mask_latent > 0.5).float()
        
        # Setup Scheduler
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.pipe.scheduler.timesteps
        
        # Start from random noise
        latents = torch.randn_like(init_latents)
        
        # Embeddings
        text_input = self.pipe.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        cond_emb = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = self.pipe.tokenizer([config.NEGATIVE_PROMPT], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        uncond_emb = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_emb = torch.cat([uncond_emb, cond_emb])

        print(f"Inpainting with Latent Blending... (Steps: {num_inference_steps})")
        
        for i, t in enumerate(tqdm(timesteps)):
            # 1. Predict Noise
            latent_input = torch.cat([latents] * 2)
            latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)
            
            # [FIX] Ensure dtypes match UNet (Float16)
            dtype = self.pipe.unet.dtype
            latent_input = latent_input.to(dtype)
            text_emb = text_emb.to(dtype)
            
            # Standard SD forward pass (4 channels)
            noise_pred = self.pipe.unet(latent_input, t, encoder_hidden_states=text_emb).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # 2. Step (Denoise)
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
            
            # 3. Latent Blending (The Core of RePaint)
            if i < len(timesteps) - 1:
                noise_timestep = timesteps[i + 1]
                
                # Create noisy version of original image at this specific timestep
                noise = torch.randn_like(init_latents)
                init_latents_noisy = self.pipe.scheduler.add_noise(init_latents, noise, noise_timestep)
                
                # Blend: Masked area gets generated content, Background gets original noisy content
                # [FIX] Use Hard Mask here!
                latents = latents * mask_latent_hard + init_latents_noisy * (1 - mask_latent_hard)

        # Decode
        latents = latents / self.pipe.vae.config.scaling_factor
        
        # [FIX] Ensure latents are in correct dtype for VAE
        latents = latents.to(self.pipe.vae.dtype)
        
        image_tensor = self.pipe.vae.decode(latents).sample
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_np = image_tensor.cpu().permute(0, 2, 3, 1).numpy()
        generated_image = Image.fromarray((image_np[0] * 255).astype(np.uint8))
        
        # Post-processing (Soft Blending)
        generated_image = generated_image.resize((orig_w, orig_h), resample=Image.LANCZOS)
        
        mask_for_blending = mask_image.resize((orig_w, orig_h), resample=Image.NEAREST)
        mask_for_blending = mask_for_blending.filter(ImageFilter.GaussianBlur(radius=5))
        
        final_image = Image.composite(generated_image, image, mask_for_blending)
        
        return final_image
