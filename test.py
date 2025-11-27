from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import requests
from io import BytesIO
import PIL
from diffusers import StableDiffusionInpaintPipeline

model_id = "Manojb/stable-diffusion-2-base"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# prompt = "a photo of futuristic city"
# image = pipe(prompt).images[0]  
    
# image.save("city.png")
def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


# img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
# mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

# init_image = download_image(img_url).resize((512, 512))
# mask_image = download_image(mask_url).resize((512, 512))

prompt = "a simpson cartoon character wearing a red hat"
image = pipe(prompt=prompt).images[0]
image.save("ss.png")