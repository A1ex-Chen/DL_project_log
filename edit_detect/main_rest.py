from PIL import Image

import torch

from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline, DDIMScheduler, DDIMInverseScheduler

caption = ""
image_path = "/workspace/research/edit_detect/real_images/white_horse.jpg"
model_id = "google/ddpm-cifar10-32"

# load model and scheduler
pipeline = DDPMPipeline.from_pretrained(model_id, safety_checker=None)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
# raw_image = Image.open(image_path).convert("RGB").resize((32, 32))
# raw_image.save("real_images/white_horse32.jpg")

raw_image = pipeline().images[0]
latents = pipeline.invert(init=raw_image).latents

# run pipeline in inference (sample random noise and denoise)
image = pipeline(init=latents).images[0]
# image = pipeline().images[0]

# save image
raw_image.save("ddpm_generated_image0.png")
image.save("ddpm_generated_image1.png")