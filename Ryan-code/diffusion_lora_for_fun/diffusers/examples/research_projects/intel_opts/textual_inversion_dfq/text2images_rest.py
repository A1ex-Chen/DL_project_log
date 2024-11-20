import argparse
import math
import os

import torch
from neural_compressor.utils.pytorch import load
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel








args = parse_args()
# Load models and create wrapper for stable diffusion
tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

pipeline = StableDiffusionPipeline.from_pretrained(
    args.pretrained_model_name_or_path, text_encoder=text_encoder, vae=vae, unet=unet, tokenizer=tokenizer
)
pipeline.safety_checker = lambda images, clip_input: (images, False)
if os.path.exists(os.path.join(args.pretrained_model_name_or_path, "best_model.pt")):
    unet = load(args.pretrained_model_name_or_path, model=unet)
    unet.eval()
    setattr(pipeline, "unet", unet)
else:
    unet = unet.to(torch.device("cuda", args.cuda_id))
pipeline = pipeline.to(unet.device)
grid, images = generate_images(pipeline, prompt=args.caption, num_images_per_prompt=args.images_num, seed=args.seed)
grid.save(os.path.join(args.pretrained_model_name_or_path, "{}.png".format("_".join(args.caption.split()))))
dirname = os.path.join(args.pretrained_model_name_or_path, "_".join(args.caption.split()))
os.makedirs(dirname, exist_ok=True)
for idx, image in enumerate(images):
    image.save(os.path.join(dirname, "{}.png".format(idx + 1)))