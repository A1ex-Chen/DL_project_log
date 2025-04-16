import inspect
from typing import List, Optional, Union

import numpy as np
import PIL.Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import PIL_INTERPOLATION, deprecate
from diffusers.utils.torch_utils import randn_tensor


EXAMPLE_DOC_STRING = """
    Examples:
        ```
        from io import BytesIO

        import requests
        import torch
        from diffusers import DiffusionPipeline
        from PIL import Image
        from transformers import CLIPFeatureExtractor, CLIPModel

        feature_extractor = CLIPFeatureExtractor.from_pretrained(
            "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
        )
        clip_model = CLIPModel.from_pretrained(
            "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", torch_dtype=torch.float16
        )


        guided_pipeline = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            # custom_pipeline="clip_guided_stable_diffusion",
            custom_pipeline="/home/njindal/diffusers/examples/community/clip_guided_stable_diffusion.py",
            clip_model=clip_model,
            feature_extractor=feature_extractor,
            torch_dtype=torch.float16,
        )
        guided_pipeline.enable_attention_slicing()
        guided_pipeline = guided_pipeline.to("cuda")

        prompt = "fantasy book cover, full moon, fantasy forest landscape, golden vector elements, fantasy magic, dark light night, intricate, elegant, sharp focus, illustration, highly detailed, digital painting, concept art, matte, art by WLOP and Artgerm and Albert Bierstadt, masterpiece"

        url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")

        image = guided_pipeline(
            prompt=prompt,
            num_inference_steps=30,
            image=init_image,
            strength=0.75,
            guidance_scale=7.5,
            clip_guidance_scale=100,
            num_cutouts=4,
            use_cutouts=False,
        ).images[0]
        display(image)
        ```
"""




class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cut_power=1.0):
        super().__init__()

        self.cut_size = cut_size
        self.cut_power = cut_power

    def forward(self, pixel_values, num_cutouts):
        sideY, sideX = pixel_values.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(num_cutouts):
            size = int(torch.rand([]) ** self.cut_power * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = pixel_values[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)







def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


class CLIPGuidedStableDiffusion(DiffusionPipeline, StableDiffusionMixin):
    """CLIP guided stable diffusion based on the amazing repo by @crowsonkb and @Jack000
    - https://github.com/Jack000/glid-3-xl
    - https://github.dev/crowsonkb/k-diffusion
    """








    @torch.enable_grad()

    @torch.no_grad()