# inpainting_pipeline.py 파일의 191번째 줄 부근을 수정합니다.
# 수정 전: image = self.pipe.vae.decode(latents.float()).sample 
# 수정 후: image = self.pipe.vae.decode(latents).sample 
# -------------------------------------------------------------

import torch
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
import numpy as np
from PIL import Image
import config
from tqdm import tqdm
from peft import PeftModel 
import torch.nn as nn 

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
        
        # --- CRITICAL FIX: 4-채널 UNet을 9-채널 Inpainting 입력용으로 확장 ---
        unet = self.pipe.unet
        if unet.conv_in.in_channels == 4:
            print("Adapting 4-channel UNet for 9-channel Inpainting input...")
            
            out_channels = unet.conv_in.out_channels
            kernel_size = unet.conv_in.kernel_size
            padding = unet.conv_in.padding
            
            new_conv_in = nn.Conv2d(
                in_channels=9, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                padding=padding, 
                bias=unet.conv_in.bias is not None 
            ).to(unet.dtype).to(device)
            
            # Initialize weights: Copy original 4 channels, zero others
            with torch.no_grad():
                new_conv_in.weight[:, :4, :, :] = unet.conv_in.weight
                new_conv_in.weight[:, 4:, :, :] = 0 # Zero init for mask/masked_image channels
                if unet.conv_in.bias is not None:
                    new_conv_in.bias = unet.conv_in.bias
            
            unet.conv_in = new_conv_in
            print(f"UNet input channels successfully adapted to {unet.conv_in.in_channels}.")
        # -------------------------------------------------------------
        
        # 2. LoRA 로드 및 병합
        if lora_weights_path:
            print(f"Loading and merging LoRA from {lora_weights_path}...")
            unet = PeftModel.from_pretrained(self.pipe.unet, lora_weights_path)
            self.pipe.unet = unet.merge_and_unload() 
        
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
    @torch.no_grad()
    def inpaint(self, image, mask_path, prompt, exemplar_image=None, num_inference_steps=50, guidance_scale=7.5, seed=None):
        
        image_size_for_mask = (512, 512)

        # 1. BBox TXT 파일을 PIL Image 마스크로 변환 로직 (오류 해결)
        if isinstance(mask_path, str) and mask_path.lower().endswith(('.txt')):
            # YOLO TXT 파일 처리: yolo_to_mask_image 함수 사용
            input_mask = yolo_to_mask_image(mask_path, image_size=image_size_for_mask)
        elif isinstance(mask_path, str) and mask_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 이미지 파일 처리 (혹시 모를 경우)
            try:
                input_mask = Image.open(mask_path).convert("L")
                input_mask = input_mask.resize(image_size_for_mask, resample=Image.NEAREST)
            except FileNotFoundError:
                print(f"Error: Mask image file not found at {mask_path}. Using all-black mask.")
                input_mask = Image.fromarray(np.zeros(image_size_for_mask, dtype=np.uint8), 'L')
        else:
            # Fallback
            print(f"Warning: Mask path {mask_path} has unknown format. Using all-black mask.")
            input_mask = Image.fromarray(np.zeros(image_size_for_mask, dtype=np.uint8), 'L')
        
        # 2. Preprocess: Resize images/mask to standard 512x512
        input_image = image.resize((512, 512), resample=Image.LANCZOS)
        
        # VAE Encode Background and Mask
        img_tensor = torch.from_numpy(np.array(input_image)).float() / 127.5 - 1.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.pipe.vae.dtype)
        
        mask_tensor = torch.from_numpy(np.array(input_mask.convert("L"))).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device, dtype=self.pipe.unet.dtype)
        
        with torch.no_grad():
            latents_bg = self.pipe.vae.encode(img_tensor).latent_dist.sample()
            latents_bg = latents_bg * self.pipe.vae.config.scaling_factor
        
        mask_latent = torch.nn.functional.interpolate(mask_tensor, size=latents_bg.shape[-2:], mode="nearest")
        
        # 3. Initial Noise (PNI or Random)
        initial_latents = None
        
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.pipe.scheduler.timesteps
        t_start = timesteps[0] 

        if exemplar_image is not None:
            print("Applying Prior Noise Initialization (PNI)...")
            
            mask_arr = np.array(input_mask.convert("L")) 
            rows = np.any(mask_arr, axis=1)
            cols = np.any(mask_arr, axis=0)
            
            if np.any(rows) and np.any(cols):
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                h, w = ymax - ymin, xmax - xmin
                
                exemplar_resized = exemplar_image.resize((w, h), resample=Image.LANCZOS)
                composite = input_image.copy()
                composite.paste(exemplar_resized, (xmin, ymin))
                
                comp_tensor = torch.from_numpy(np.array(composite)).float() / 127.5 - 1.0
                comp_tensor = comp_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.pipe.vae.dtype)
                
                with torch.no_grad():
                    latents_comp = self.pipe.vae.encode(comp_tensor).latent_dist.sample()
                    latents_comp = latents_comp * self.pipe.vae.config.scaling_factor
                
                noise = torch.randn(latents_comp.shape, device=self.device, dtype=latents_comp.dtype, generator=None)
                initial_latents = self.pipe.scheduler.add_noise(latents_comp, noise, t_start) 
            else:
                print("Warning: Mask is empty. Skipping PNI and using random noise.")
        
        if initial_latents is None:
            initial_latents = torch.randn(latents_bg.shape, device=self.device, dtype=latents_bg.dtype, generator=None)
            
        # 4. Text Embeddings
        text_input = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
        
        uncond_input = self.pipe.tokenizer(
            [config.NEGATIVE_PROMPT], padding="max_length", max_length=text_input.input_ids.shape[-1], return_tensors="pt"
        )
        uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # 5. Prepare Inpainting Condition (Masked Image Latent + Mask)
        masked_image_latent = latents_bg * (1 - mask_latent) 
        latents = initial_latents

        self.pipe.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.pipe.scheduler.timesteps
        
        # 6. Denoising Loop (PNI Latent 사용 및 Inpainting Blending)
        print("Inpainting...")
        for t in tqdm(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            
            # UNet Input: [noisy_latent_4, masked_image_latent_4, mask_1] = (B, 9, H, W)
            unet_input = torch.cat([latent_model_input, masked_image_latent.repeat(2, 1, 1, 1), mask_latent.repeat(2, 1, 1, 1)], dim=1)
            
            noise_pred = self.pipe.unet(unet_input, t, encoder_hidden_states=text_embeddings).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
            latents = latents * mask_latent + latents_bg * (1 - mask_latent)

        # latents = latents * mask_latent + latents_bg * (1 - mask_latent)
        # 7. Decode
        latents = 1 / self.pipe.vae.config.scaling_factor * latents
        # FIX: .float() 제거
        image = self.pipe.vae.decode(latents).sample 
        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        
        return Image.fromarray(image[0])