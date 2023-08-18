# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import torch

from diffusers import DiffusionPipeline
from torch import autocast
# -

# # Settings

PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"
DEVICE = "cpu"

# # Model

model = DiffusionPipeline.from_pretrained(PRETRAINED_MODEL, safety_checker = None)

model.to(DEVICE)
model.enable_attention_slicing()


# # Generate image

def generate_image(prompt, device=DEVICE):
    with autocast(DEVICE):
        output = model(prompt) 
    image = output.images[0]
    image.show()


# # Demo

generate_image("An image of a mammoth with cinematic light")
