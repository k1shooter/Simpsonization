# 파일: inpainting_pipeline.py
# (상단 import 및 yolo_to_mask_image 함수는 기존과 동일하게 유지)
import torch
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler, UNet2DConditionModel
import numpy as np
from PIL import Image, ImageFilter
import config
from peft import PeftModel 
import torch.nn as nn 
import os
from tqdm import tqdm   
# ... (yolo_to_mask_image 함수 생략 - 이전과 동일)
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
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(img_w, x_max), min(img_h, y_max)
            mask_arr[y_min:y_max, x_min:x_max] = 255
    except Exception:
        pass
    return Image.fromarray(mask_arr, 'L')

class CustomInpaintingPipeline:
    def __init__(self, base_model_id, lora_weights_path, device="cuda"):
        self.device = device
        print(f"Loading Pipeline: {base_model_id}")
        
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            base_model_id, 
            torch_dtype=torch.float16,
            safety_checker=None
        )
        self.pipe.to(device)
        
        # 4ch -> 9ch 확장
        if self.pipe.unet.config.in_channels == 4:
            print("Adapting UNet 4ch -> 9ch...")
            with torch.no_grad():
                new_config = self.pipe.unet.config
                new_config["in_channels"] = 9
                new_unet = UNet2DConditionModel.from_config(new_config).to(device, dtype=self.pipe.unet.dtype)
                
                state_dict = self.pipe.unet.state_dict()
                if "conv_in.weight" in state_dict: del state_dict["conv_in.weight"]
                new_unet.load_state_dict(state_dict, strict=False)
                
                old_conv = self.pipe.unet.conv_in
                new_conv = new_unet.conv_in
                new_conv.weight[:, :4, :, :] = old_conv.weight
                new_conv.weight[:, 4:, :, :] = 0 
                if old_conv.bias is not None: new_conv.bias = old_conv.bias
                
                self.pipe.unet = new_unet

        # LoRA 로드
        if lora_weights_path and os.path.exists(lora_weights_path):
            try:
                self.pipe.unet = PeftModel.from_pretrained(self.pipe.unet, lora_weights_path)
                self.pipe.unet = self.pipe.unet.merge_and_unload()
                print("LoRA merged successfully.")
            except Exception as e:
                print(f"Error loading LoRA: {e}")
        
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
    
    def prepare_mask_and_masked_image(self, image, mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        
        mask = np.array(mask.convert("L")).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        
        masked_image = image * (1 - mask)
        return mask, masked_image, image 

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
        
        # Tensors
        mask_tensor, masked_image_tensor, orig_image_tensor = self.prepare_mask_and_masked_image(input_image, mask_image)
        mask_tensor = mask_tensor.to(self.device, dtype=self.pipe.unet.dtype)
        masked_image_tensor = masked_image_tensor.to(self.device, dtype=self.pipe.unet.dtype)
        orig_image_tensor = orig_image_tensor.to(self.device, dtype=self.pipe.unet.dtype)

        # Latents
        init_latents = self.pipe.vae.encode(orig_image_tensor).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        masked_image_latents = self.pipe.vae.encode(masked_image_tensor).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        mask_latent = torch.nn.functional.interpolate(mask_tensor, size=init_latents.shape[-2:], mode="nearest")
        
        # PNI (Original Face + Noise)
        noise = torch.randn_like(init_latents)
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        
        init_timestep = int(num_inference_steps * strength)
        timesteps = self.pipe.scheduler.timesteps[-init_timestep:]
        timesteps = timesteps.to(self.device)
        timesteps = torch.cat([timesteps, torch.tensor([0], device=self.device).long()])
        
        latents = self.pipe.scheduler.add_noise(init_latents, noise, timesteps[0])

        # Embeddings
        text_input = self.pipe.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        cond_emb = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = self.pipe.tokenizer([config.NEGATIVE_PROMPT], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        uncond_emb = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_emb = torch.cat([uncond_emb, cond_emb])

        print(f"Inpainting... (Strength: {strength}, Guidance: {guidance_scale})")
        for t in tqdm(timesteps[:-1]):
            latent_input = torch.cat([latents] * 2)
            latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)
            
            unet_input = torch.cat([latent_input, masked_image_latents.repeat(2,1,1,1), mask_latent.repeat(2,1,1,1)], dim=1)
            
            noise_pred = self.pipe.unet(unet_input, t, encoder_hidden_states=text_emb).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
            
            # [Restored] Latent Blending to enforce background consistency
            if t > timesteps[-1]:
                # We need the noisy background at the *next* timestep (t_prev)
                # Since we just stepped from t -> t_prev, 'latents' is at t_prev.
                # We need to add noise to init_latents (clean bg) to match t_prev.
                
                # Find t_prev (the timestep we just stepped TO)
                # In the loop, 't' is the current step. step() moves to the next.
                # We can approximate or calculate it.
                # Simpler approach: Just use the noise level of 't' if the step size is small, 
                # but for correctness let's use the scheduler's logic or just re-noise.
                
                # Actually, since we are using 'timesteps' array which is ordered from high to low:
                # The loop variable 't' is where we started.
                # We need to know where we ended up.
                
                # Let's find index of t
                idx = (timesteps == t).nonzero()
                if len(idx) > 0:
                    idx = idx[0].item()
                    if idx < len(timesteps) - 1:
                        t_prev = timesteps[idx + 1]
                        
                        # Add noise to clean background to reach t_prev
                        noise_bg = torch.randn_like(init_latents)
                        latents_bg_noisy = self.pipe.scheduler.add_noise(init_latents, noise_bg, t_prev)
                        
                        # Blend
                        latents = latents * mask_latent + latents_bg_noisy * (1 - mask_latent)

        # Decode
        latents = latents / self.pipe.vae.config.scaling_factor
        image_tensor = self.pipe.vae.decode(latents).sample
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_np = image_tensor.cpu().permute(0, 2, 3, 1).numpy()
        generated_image = Image.fromarray((image_np[0] * 255).astype(np.uint8))
        
        # [핵심 FIX] Soft Blending
        generated_image = generated_image.resize((orig_w, orig_h), resample=Image.LANCZOS)
        
        # 마스크를 부드럽게 블러링 (Soft Mask)
        mask_for_blending = mask_image.resize((orig_w, orig_h), resample=Image.NEAREST)
        mask_for_blending = mask_for_blending.filter(ImageFilter.GaussianBlur(radius=5)) # 경계 흐리기
        
        # 부드러운 마스크를 이용해 합성
        final_image = Image.composite(generated_image, image, mask_for_blending)
        
        return final_image