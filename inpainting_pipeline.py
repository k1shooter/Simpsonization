import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np
from PIL import Image
import config
from tqdm import tqdm

class CustomInpaintingPipeline:
    def __init__(self, base_model_id, lora_weights_path, device="cuda"):
        self.device = device
        print(f"Loading pipeline from {base_model_id}...")
        self.pipe = StableDiffusionPipeline.from_pretrained(base_model_id)
        self.pipe.to(device)
        
        # Load LoRA
        if lora_weights_path:
            print(f"Loading LoRA from {lora_weights_path}...")
            from peft import PeftModel
            self.pipe.unet = PeftModel.from_pretrained(self.pipe.unet, lora_weights_path)
            self.pipe.unet.merge_and_unload() # Merge for speed and simplicity
        
        # Use DDIM for deterministic/faster sampling
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
    @torch.no_grad()
    def inpaint(self, image, mask, prompt, exemplar_image=None, num_inference_steps=50, guidance_scale=7.5, seed=None):
        """
        image: PIL Image (Background)
        mask: PIL Image (Mask, white is inpainting area)
        exemplar_image: PIL Image (Optional, for Prior Noise Initialization)
        """
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)
        else:
            generator = None
            
        # 1. Preprocess
        width, height = image.size
        # Resize to 512x512
        image = image.resize((512, 512), resample=Image.LANCZOS)
        mask = mask.resize((512, 512), resample=Image.NEAREST)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        mask_tensor = torch.from_numpy(np.array(mask.convert("L"))).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        mask_tensor = (mask_tensor > 0.5).float()
        
        # Encode image to latent
        latents_bg = self.pipe.vae.encode(img_tensor).latent_dist.sample()
        latents_bg = latents_bg * self.pipe.vae.config.scaling_factor
        
        # Resize mask to latent size
        mask_latent = torch.nn.functional.interpolate(mask_tensor, size=latents_bg.shape[-2:], mode="nearest")
        
        # 2. Prepare Timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.pipe.scheduler.timesteps
        
        # 3. Initial Noise (PNI or Random)
        if exemplar_image is not None:
            print("Applying Prior Noise Initialization (PNI)...")
            # Create composite image
            # Find bbox of mask
            mask_arr = np.array(mask.convert("L"))
            rows = np.any(mask_arr, axis=1)
            cols = np.any(mask_arr, axis=0)
            if not np.any(rows) or not np.any(cols):
                # Empty mask?
                print("Warning: Empty mask, skipping PNI.")
                latents = torch.randn(latents_bg.shape, device=self.device, dtype=latents_bg.dtype, generator=generator)
            else:
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                h, w = ymax - ymin, xmax - xmin
                
                # Resize exemplar to fit bbox
                exemplar_resized = exemplar_image.resize((w, h), resample=Image.LANCZOS)
                
                # Paste
                composite = image.copy()
                composite.paste(exemplar_resized, (xmin, ymin))
                
                # Encode composite
                comp_tensor = torch.from_numpy(np.array(composite)).float() / 127.5 - 1.0
                comp_tensor = comp_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                latents_comp = self.pipe.vae.encode(comp_tensor).latent_dist.sample()
                latents_comp = latents_comp * self.pipe.vae.config.scaling_factor
                
                # Add noise to reach T
                noise = torch.randn(latents_comp.shape, device=self.device, dtype=latents_comp.dtype, generator=generator)
                latents = self.pipe.scheduler.add_noise(latents_comp, noise, timesteps[0:1])
        else:
            latents = torch.randn(latents_bg.shape, device=self.device, dtype=latents_bg.dtype, generator=generator)
        
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
            [""], padding="max_length", max_length=text_input.input_ids.shape[-1], return_tensors="pt"
        )
        uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # 5. Denoising Loop
        print("Inpainting...")
        for t in tqdm(timesteps):
            # Expand latents for classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            # Guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Step
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Blending (Masked Diffusion)
            if t > timesteps[-1]: 
                step_index = (timesteps == t).nonzero().item()
                if step_index < len(timesteps) - 1:
                    t_prev = timesteps[step_index + 1]
                    
                    noise_bg = torch.randn(latents_bg.shape, device=self.device, dtype=latents_bg.dtype, generator=generator)
                    latents_bg_noisy = self.pipe.scheduler.add_noise(latents_bg, noise_bg, t_prev)
                    
                    # Blend
                    latents = latents * mask_latent + latents_bg_noisy * (1 - mask_latent)
                else:
                    latents = latents * mask_latent + latents_bg * (1 - mask_latent)

        # 6. Decode
        latents = 1 / self.pipe.vae.config.scaling_factor * latents
        image = self.pipe.vae.decode(latents).sample
        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        
        return Image.fromarray(image[0])

if __name__ == "__main__":
    # Test
    pass
