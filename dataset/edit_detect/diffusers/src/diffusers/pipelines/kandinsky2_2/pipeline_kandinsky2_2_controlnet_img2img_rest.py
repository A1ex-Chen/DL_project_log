# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from PIL import Image

from ...models import UNet2DConditionModel, VQModel
from ...schedulers import DDPMScheduler
from ...utils import (
    logging,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> import numpy as np

        >>> from diffusers import KandinskyV22PriorEmb2EmbPipeline, KandinskyV22ControlnetImg2ImgPipeline
        >>> from transformers import pipeline
        >>> from diffusers.utils import load_image


        >>> def make_hint(image, depth_estimator):
        ...     image = depth_estimator(image)["depth"]
        ...     image = np.array(image)
        ...     image = image[:, :, None]
        ...     image = np.concatenate([image, image, image], axis=2)
        ...     detected_map = torch.from_numpy(image).float() / 255.0
        ...     hint = detected_map.permute(2, 0, 1)
        ...     return hint


        >>> depth_estimator = pipeline("depth-estimation")

        >>> pipe_prior = KandinskyV22PriorEmb2EmbPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
        ... )
        >>> pipe_prior = pipe_prior.to("cuda")

        >>> pipe = KandinskyV22ControlnetImg2ImgPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-controlnet-depth", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> img = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/cat.png"
        ... ).resize((768, 768))


        >>> hint = make_hint(img, depth_estimator).unsqueeze(0).half().to("cuda")

        >>> prompt = "A robot, 4k photo"
        >>> negative_prior_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

        >>> generator = torch.Generator(device="cuda").manual_seed(43)

        >>> img_emb = pipe_prior(prompt=prompt, image=img, strength=0.85, generator=generator)
        >>> negative_emb = pipe_prior(prompt=negative_prior_prompt, image=img, strength=1, generator=generator)

        >>> images = pipe(
        ...     image=img,
        ...     strength=0.5,
        ...     image_embeds=img_emb.image_embeds,
        ...     negative_image_embeds=negative_emb.image_embeds,
        ...     hint=hint,
        ...     num_inference_steps=50,
        ...     generator=generator,
        ...     height=768,
        ...     width=768,
        ... ).images

        >>> images[0].save("robot_cat.png")
        ```
"""


# Copied from diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2.downscale_height_and_width


# Copied from diffusers.pipelines.kandinsky.pipeline_kandinsky_img2img.prepare_image


class KandinskyV22ControlnetImg2ImgPipeline(DiffusionPipeline):
    """
    Pipeline for image-to-image generation using Kandinsky

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with `unet` to generate image latents.
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.
        movq ([`VQModel`]):
            MoVQ Decoder to generate the image from the latents.
    """

    model_cpu_offload_seq = "unet->movq"


    # Copied from diffusers.pipelines.kandinsky.pipeline_kandinsky_img2img.KandinskyImg2ImgPipeline.get_timesteps

    # Copied from diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2_img2img.KandinskyV22Img2ImgPipeline.prepare_latents

    @torch.no_grad()