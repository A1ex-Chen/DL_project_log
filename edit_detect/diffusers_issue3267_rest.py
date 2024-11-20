import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPix2PixZeroPipeline

import requests
from PIL import Image

captioner_id = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(captioner_id)
model = BlipForConditionalGeneration.from_pretrained(
    captioner_id, low_cpu_mem_usage=True
)

sd_model_ckpt = 'runwayml/stable-diffusion-v1-5'

pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(
    sd_model_ckpt,
    caption_generator=model,
    caption_processor=processor,
    safety_checker=None,
)

pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()

img_url = "https://github.com/pix2pixzero/pix2pix-zero/raw/main/assets/test_images/cats/cat_6.png"

raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB").resize((512, 512))
raw_image.save("diffusers_issue3267_raw_image.jpg")
# generate caption
caption = pipeline.generate_caption(raw_image)

inv_latents = pipeline.invert(caption, image=raw_image).latents

import torch
from diffusers import StableDiffusionPipeline

sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_ckpt, scheduler=pipeline.scheduler)
sd_pipe = sd_pipe.to("cuda")

image = sd_pipe(caption, latents=inv_latents).images[0]
image.save("diffusers_issue3267_image.jpg")