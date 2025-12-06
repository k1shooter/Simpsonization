# Vis Contest: Simpson Face Inpainting Project

“Simpsonize” a real face into a 3D-looking Simpson character while keeping the person’s structure and background intact. The pipeline masks the face, re-imagines it with yellow skin, big round eyes, and claymation/Pixar lighting, then blends the generated face back into the original photo.

## 🚧 Project Status: `Inpainting Ver2` (Hybrid, active)
- 9-channel UNet input for context: noisy latent + masked latent + mask so the model sees background/lighting.
- Repaint-style latent blending to preserve background and prevent artifacts.
- ControlNet (Canny) guidance during inference to keep facial layout.
- LoRA fine-tuning on `to_k`/`to_v` only, keeping base geometry stable.

### Key improvements in Ver2
- Synchronized augmentation: image/mask flips stay aligned.
- Rounded-rectangle masks for better face coverage.
- Hard masking in the latent loop to stop background noise bleeding into the face.

## 📂 Directory Structure
- `inpainting_ver2/` (active)
  - `train_lora_v2.py`: Train the hybrid LoRA (9-ch UNet + masked loss).
  - `inpainting_pipeline_v2.py`: Hybrid pipeline (9-ch + repaint logic).
  - `run_inpainting_v2.py`: Single-image inference entrypoint.
  - `config_v2.py`: Hyperparams, paths, prompts (default output dir: `lora_weights_v2`).
- `repaint_workspace/`: Older 4-ch repaint attempt (kept for reference).
- `train_lora.py`, `inpainting_pipeline.py`: Original v1 files.

## 🔬 How It Works (conceptual)
1) Mask and context: Input face image + binary/soft mask isolate the face; background stays visible to guide lighting and geometry.
2) 9-channel UNet: Concatenate noisy latent, masked latent, and mask to give the model context and a precise inpaint target.
3) Masked denoising loss: Train only on masked regions, forcing the model to focus on face reconstruction.
4) ControlNet (Canny): At inference, edge cues keep pose and layout stable.
5) Repaint loop: Re-inject original-noise latents outside the mask each step to preserve background.

## 🚀 Quickstart

### 1) Train LoRA (Hybrid)
```bash
python3 inpainting_ver2/train_lora_v2.py
```
- Output: `lora_weights_v2`
- Adjust `inpainting_ver2/config_v2.py` for paths (`SIMPSON_FACE_ROOT`, `SIMPSON_MASK_ROOT`), prompt, LR, epochs, LoRA rank/alpha.

### 2) Single-image Inference
```bash
python3 inpainting_ver2/run_inpainting_v2.py \
  --image /path/to/image.jpg \
  --mask /path/to/mask.txt \
  --output output_v2.png \
  --guidance_scale 16.0 \
  --strength 0.75
```
- Mask can be YOLO `.txt` (auto-converted) or an image mask.

### 3) Batch Inference (auto mask + inpaint, recommended)
```bash
python3 run_batch_explicit.py \
  --input_dir /path/to/val_images \
  --label_dir /path/to/yolo_labels \
  --output_dir batch_results_explicit \
  --num_samples 10 \
  --control_strength 0.6
```
- Steps: create YOLO mask → run ControlNet inpainting → cleanup temp masks.

## ⚙️ Key Settings (config_v2.py)
- Model: `MODEL_ID` (default SD2-base), device auto-select.
- LoRA: `LORA_RANK`, `LORA_ALPHA`, `TARGET_MODULES=["to_k","to_v"]`.
- Training: `TRAIN_EPOCHS`, `LEARNING_RATE`, `OUTPUT_DIR`.
- Data: `SIMPSON_FACE_ROOT`, `SIMPSON_MASK_ROOT` (paired masks), `EXEMPLAR_PROMPT` (3D-heavy prompt).
- Inference: `NUM_INFERENCE_STEPS`, `GUIDANCE_SCALE`, `NEGATIVE_PROMPT`.

## 🧪 Tips for Quality
- Data/masks: ensure masks tightly cover the face; poor masks hurt both loss and blending.
- Prompting: keep 3D/lighting cues; use strong negative prompt to avoid flat/2D outputs.
- Control strength: raise to lock pose/edges; lower if over-constrained.
- Strength (denoise): higher = bigger style change, lower = more original detail.
- Resources: 9-ch UNet + ControlNet is VRAM-heavy; reduce batch size or dtype if needed.

## 🔭 Future Improvements
- Train on 3D-rendered Simpson faces (SDXL or 3D assets) to strengthen geometry.
- Hi-res pass or tiling for higher fidelity after 512x512 generation.
- Additional ControlNets (depth/normal) for better 3D consistency.
