from diffusers import AutoPipelineForInpainting, DEISMultistepScheduler
import torch
from diffusers.utils import load_image

# Inpainting modeli için pipeline'ı yükle
pipe = AutoPipelineForInpainting.from_pretrained('lykon-models/dreamshaper-8-inpainting', torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")


image_path = "insan.jpg"  # Resim dosyanızın yolu
mask_image_path = "output.png"  # Maske dosyanızın yolu

# Resmi ve maskeyi yükle
image = load_image(image_path)
mask_image = load_image(mask_image_path)

# Inpainting işlemi için prompt belirleyin
prompt = "a majestic tiger sitting on a park bench"

# İşlemi tekrarlanabilir hale getirmek için bir rastgelelik tohumu belirleyin
generator = torch.manual_seed(33)

# Inpainting işlemini gerçekleştirin
output_image = pipe(prompt, image=image, mask_image=mask_image, generator=generator, num_inference_steps=25).images[0]

# Sonucu kaydedin
output_image.save("./image.png")