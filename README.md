# Vis Contest: Simpson Face Inpainting Project

This project aims to replace a masked face in an image with a **3D Simpson-style face** using Stable Diffusion and LoRA.

## 🚧 Project Status: `Inpainting Ver2` (Hybrid)

We are currently in the **Hybrid Phase**, combining the best of both worlds:
1.  **9-Channel Input (Context Awareness)**: From the original inpainting approach. The model sees the background to understand lighting and geometry (crucial for 3D effect).
2.  **Repaint Logic (Background Preservation)**: From the Repaint approach. We enforce background consistency during the denoising loop to prevent artifacts.

### Key Improvements in Ver2
*   **Synchronized Data Augmentation**: Fixed a critical bug where images were flipped but masks were not during training.
*   **Rounded Rectangle Mask**: Replaced the simple box/ellipse mask with a rounded rectangle for better face coverage.
*   **Hard Masking**: Used hard masking in the latent loop to prevent background noise from bleeding into the generated face.

## 📂 Directory Structure

*   **`inpainting_ver2/` (Current Active Workspace)**:
    *   `train_lora_v2.py`: **[Action Required]** Run this to train the Hybrid LoRA.
    *   `inpainting_pipeline_v2.py`: Implements the Hybrid pipeline (9-channel + Repaint logic).
    *   `run_inpainting_v2.py`: Inference script using the Hybrid pipeline.
    *   `config_v2.py`: Configuration for Ver2 (Output dir: `lora_weights_v2`).
*   `repaint_workspace/`: Previous attempt using standard 4-channel SD + Repaint. (Deprecated for 3D quality, but logic was useful).
*   `train_lora.py`, `inpainting_pipeline.py`: Original v1 files.

## 🚀 How to Run

### 1. Train LoRA (Hybrid)
This is the **immediate next step**. We need to train the model with the fixed augmentation and 9-channel architecture.

```bash
# In root directory
/root/anaconda3/envs/vis_con/bin/python inpainting_ver2/train_lora_v2.py
```
*   **Output**: Weights will be saved to `lora_weights_v2`.
*   **Log**: Check `train_log_v2.txt` (if running in background).

### 2. Inference (Generate Images)
After training is complete:

```bash
/root/anaconda3/envs/vis_con/bin/python inpainting_ver2/run_inpainting_v2.py \
  --image /path/to/image.jpg \
  --mask /path/to/mask.txt \
  --output output_v2.png
```

## 💡 Future Improvements (If 3D quality is still low)
If the current Hybrid approach still yields 2D-looking results, it is likely due to the **2D nature of the training dataset**.
*   **Action**: Replace the training dataset with **3D rendered Simpson images** (generated via SDXL or obtained from 3D assets).
*   **Why**: The model needs to learn "Geometry", not just "Style".

## 📝 Configuration
Check `inpainting_ver2/config_v2.py` for:
*   `SIMPSON_FACE_ROOT`: Path to training images.
*   `EXEMPLAR_PROMPT`: Prompt used for training (contains 3D keywords).
*   `OUTPUT_DIR`: Where LoRA weights are saved.
