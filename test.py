import torch
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler, UNet2DConditionModel
import numpy as np
from PIL import Image, ImageFilter
import os

# --- 설정 (이전과 동일하게 맞춤) ---
MODEL_ID = "Manojb/stable-diffusion-2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT = "a 3d render of a simpson face, claymation, volumetric lighting, 3d modeling, unreal engine, high fidelity, 8k, detailed texture"
NEGATIVE_PROMPT = "2d, flat, drawing, sketch, paper, cartoon, anime, low quality, distorted, blurry, bad anatomy"
GUIDANCE_SCALE = 15.0
NUM_INFERENCE_STEPS = 50
STRENGTH = 0.75 # PNI와 유사하게 원본 구조를 유지하며 생성하는 강도

# 테스트할 이미지 및 라벨 경로
IMAGE_PATH = "/root/vis_contest/data/datasets/fareselmenshawii/face-detection-dataset/versions/3/images/train/0a36708ffa83d9f6.jpg"
MASK_PATH = "/root/vis_contest/data/datasets/fareselmenshawii/face-detection-dataset/versions/3/labels/train/0a36708ffa83d9f6.txt"

# --- 유틸리티: YOLO 마스크 변환 ---
def yolo_to_mask_image(txt_path, image_size=(512, 512)):
    img_w, img_h = image_size
    mask_arr = np.zeros(image_size, dtype=np.uint8)
    try:
        if os.path.exists(txt_path):
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
        else:
            print(f"Mask file not found: {txt_path}")
    except Exception as e:
        print(f"Error loading mask: {e}")
    return Image.fromarray(mask_arr, 'L')

def main():
    print(f"Loading Baseline Model: {MODEL_ID}")
    
    # 1. Pipeline 로드 (Standard Inpaint Pipeline)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(DEVICE)

    # 2. [필수] UNet 채널 확장 (4ch -> 9ch)
    # T2I 모델을 인페인팅 파이프라인에서 돌리려면 입력 채널을 맞춰줘야 함
    if pipe.unet.config.in_channels == 4:
        print("Adapting UNet 4ch -> 9ch for Inpainting compatibility...")
        with torch.no_grad():
            new_config = pipe.unet.config
            new_config["in_channels"] = 9
            new_unet = UNet2DConditionModel.from_config(new_config).to(DEVICE, dtype=pipe.unet.dtype)
            
            # 기존 가중치 로드 (conv_in 제외)
            state_dict = pipe.unet.state_dict()
            if "conv_in.weight" in state_dict: del state_dict["conv_in.weight"]
            new_unet.load_state_dict(state_dict, strict=False)
            
            # conv_in 가중치 이식 (기존 4채널 복사, 나머지 0)
            old_conv = pipe.unet.conv_in
            new_conv = new_unet.conv_in
            new_conv.weight[:, :4, :, :] = old_conv.weight
            new_conv.weight[:, 4:, :, :] = 0 
            if old_conv.bias is not None: new_conv.bias = old_conv.bias
            
            pipe.unet = new_unet
            print("UNet adapted.")

    # 3. 데이터 로드 및 전처리
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        return

    print(f"Processing: {os.path.basename(IMAGE_PATH)}")
    original_image = Image.open(IMAGE_PATH).convert("RGB")
    orig_w, orig_h = original_image.size
    
    # 모델 입력용 리사이즈
    input_image = original_image.resize((512, 512), Image.LANCZOS)
    
    # 마스크 로드 (YOLO txt -> Image)
    if MASK_PATH.endswith(".txt"):
        mask_image = yolo_to_mask_image(MASK_PATH, (512, 512))
    else:
        mask_image = Image.open(MASK_PATH).convert("L").resize((512, 512), Image.NEAREST)

    # 4. 추론 실행 (LoRA 없음)
    # strength < 1.0을 주면, 입력 이미지를 시작점으로 사용하여(PNI 효과) 구조를 유지하려 시도합니다.
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=input_image,
        mask_image=mask_image,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        strength=STRENGTH 
    ).images[0]

    # 5. [핵심] 배경 강제 복구 (Pixel-level Composite)
    # 생성된 이미지가 배경을 미세하게 변형시키는 것을 방지하기 위해 원본 배경을 덮어씌웁니다.
    
    # 생성된 이미지를 원본 크기로 복원
    result_resized = result.resize((orig_w, orig_h), Image.LANCZOS)
    
    # 마스크도 원본 크기로 복원 (부드러운 합성을 위해 블러링 적용)
    mask_resized = mask_image.resize((orig_w, orig_h), Image.NEAREST)
    mask_soft = mask_resized.filter(ImageFilter.GaussianBlur(radius=5))
    
    # 원본 이미지 위에 생성된 얼굴만 합성
    final_image = Image.composite(result_resized, original_image, mask_soft)

    # 6. 저장
    output_filename = "baseline_no_lora.png"
    final_image.save(output_filename)
    print(f"Saved baseline result to {output_filename}")

if __name__ == "__main__":
    main()