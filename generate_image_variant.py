# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + id="5f30ee1c-ec67-4b59-b88d-5e98b60a85df"
import torch

from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
from PIL import Image, ImageOps
from torch import autocast

# + [markdown] id="624a04ba-3aa1-49b1-8f4a-aaa097efa7f6"
# # Settings

# + id="84f61a71-827f-441e-827f-6c7d8a3af254"
PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"

# + id="00290292-aa03-4c1a-9c5c-18bd0f8f6aea"
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
DEVICE

# + [markdown] id="860ba9d8-0ce5-4f6c-922e-e3b326fbd112"
# # Model

# + id="4e03e78a-927c-409c-8d91-0d831ed5a181"
scheduler = DDIMScheduler(beta_start = 0.0005,
                          beta_schedule = "scaled_linear",
                          clip_sample = False,
                          set_alpha_to_one = False)

model = StableDiffusionImg2ImgPipeline.from_pretrained(PRETRAINED_MODEL,
                                                       scheduler = scheduler,
                                                       revision = "fp16",
                                                       torch_type = torch.float16)

generator = torch.Generator(device = DEVICE).manual_seed(31)

# + id="814ad28a-ea04-43d5-8e74-5d26c59a83ca"
model.to(DEVICE)
model.enable_attention_slicing()


# + [markdown] id="29d847e8-903c-4686-a51b-f32f5fe1fa18"
# # Generate image

# + id="xU9CzDmodig5"
def load_image(image_path, size=500):
    size = size - size % 32
    image = Image.open(image_path)
    return ImageOps.contain(image, (size, size)).convert("RGB")


# + id="f346c54f-79ca-41af-8b25-394aab1314ed"
def generate_image_variant(prompt, image_path="resources/sample_image.jpg",
                           num_inference_steps=50, strength=0.6, guidance_scale=7.5,
                           device=DEVICE):
    input_image = load_image(image_path)
    with autocast(DEVICE):
        output = model(
        prompt = prompt,
        image = input_image,
        num_inference_steps = num_inference_steps,
        strength = strength,
        guidance_scale = guidance_scale,
        generator = generator)
    variant = output.images[0]
    variant.show()
    display(variant)


# + [markdown] id="91ccc291-f630-4fff-b086-e80d71e3ca17"
# # Demo

# + id="e5e42c1c-a461-4a40-ab30-17faf12c2c55"
prompt = "hot chocolate with marshmallow"
image = "resources/sample_image.jpg"
generate_image_variant(prompt, image)
