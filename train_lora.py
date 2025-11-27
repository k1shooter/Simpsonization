import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torchvision import transforms
import config
import os
from tqdm import tqdm

def train_lora():
    print(f"Loading model: {config.MODEL_ID}")
    pipeline = StableDiffusionPipeline.from_pretrained(config.MODEL_ID)
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
        # Fallback to dataset loading
        from prepare_data import get_exemplar
        image = get_exemplar()
        
    # Transform
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    pixel_values = transform(image).unsqueeze(0).to(config.DEVICE)
    
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
        latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * pipeline.vae.config.scaling_factor
        
        # 2. Sample Noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=config.DEVICE)
        timesteps = timesteps.long()
        
        # 3. Add Noise
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 4. Predict Noise
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
        # 5. Loss
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
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
