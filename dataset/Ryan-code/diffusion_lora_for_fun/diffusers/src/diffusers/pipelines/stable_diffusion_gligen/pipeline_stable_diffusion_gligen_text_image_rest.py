# Copyright 2024 The GLIGEN Authors and HuggingFace Team. All rights reserved.
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

import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import PIL.Image
import torch
from transformers import (
    CLIPFeatureExtractor,
    CLIPProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from ...image_processor import VaeImageProcessor
from ...loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, UNet2DConditionModel
from ...models.attention import GatedSelfAttentionDense
from ...models.lora import adjust_lora_scale_text_encoder
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import USE_PEFT_BACKEND, logging, replace_example_docstring, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from ..stable_diffusion import StableDiffusionPipelineOutput
from ..stable_diffusion.clip_image_project_model import CLIPImageProjection
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionGLIGENTextImagePipeline
        >>> from diffusers.utils import load_image

        >>> # Insert objects described by image at the region defined by bounding boxes
        >>> pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
        ...     "anhnct/Gligen_Inpainting_Text_Image", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> input_image = load_image(
        ...     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/livingroom_modern.png"
        ... )
        >>> prompt = "a backpack"
        >>> boxes = [[0.2676, 0.4088, 0.4773, 0.7183]]
        >>> phrases = None
        >>> gligen_image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/backpack.jpeg"
        ... )

        >>> images = pipe(
        ...     prompt=prompt,
        ...     gligen_phrases=phrases,
        ...     gligen_inpaint_image=input_image,
        ...     gligen_boxes=boxes,
        ...     gligen_images=[gligen_image],
        ...     gligen_scheduled_sampling_beta=1,
        ...     output_type="pil",
        ...     num_inference_steps=50,
        ... ).images

        >>> images[0].save("./gligen-inpainting-text-image-box.jpg")

        >>> # Generate an image described by the prompt and
        >>> # insert objects described by text and image at the region defined by bounding boxes
        >>> pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
        ...     "anhnct/Gligen_Text_Image", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a flower sitting on the beach"
        >>> boxes = [[0.0, 0.09, 0.53, 0.76]]
        >>> phrases = ["flower"]
        >>> gligen_image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/pexels-pixabay-60597.jpg"
        ... )

        >>> images = pipe(
        ...     prompt=prompt,
        ...     gligen_phrases=phrases,
        ...     gligen_images=[gligen_image],
        ...     gligen_boxes=boxes,
        ...     gligen_scheduled_sampling_beta=1,
        ...     output_type="pil",
        ...     num_inference_steps=50,
        ... ).images

        >>> images[0].save("./gligen-generation-text-image-box.jpg")

        >>> # Generate an image described by the prompt and
        >>> # transfer style described by image at the region defined by bounding boxes
        >>> pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
        ...     "anhnct/Gligen_Text_Image", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a dragon flying on the sky"
        >>> boxes = [[0.4, 0.2, 1.0, 0.8], [0.0, 1.0, 0.0, 1.0]]  # Set `[0.0, 1.0, 0.0, 1.0]` for the style

        >>> gligen_image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png"
        ... )

        >>> gligen_placeholder = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png"
        ... )

        >>> images = pipe(
        ...     prompt=prompt,
        ...     gligen_phrases=[
        ...         "dragon",
        ...         "placeholder",
        ...     ],  # Can use any text instead of `placeholder` token, because we will use mask here
        ...     gligen_images=[
        ...         gligen_placeholder,
        ...         gligen_image,
        ...     ],  # Can use any image in gligen_placeholder, because we will use mask here
        ...     input_phrases_mask=[1, 0],  # Set 0 for the placeholder token
        ...     input_images_mask=[0, 1],  # Set 0 for the placeholder image
        ...     gligen_boxes=boxes,
        ...     gligen_scheduled_sampling_beta=1,
        ...     output_type="pil",
        ...     num_inference_steps=50,
        ... ).images

        >>> images[0].save("./gligen-generation-text-image-box-style-transfer.jpg")
        ```
"""


class StableDiffusionGLIGENTextImagePipeline(DiffusionPipeline, StableDiffusionMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion with Grounded-Language-to-Image Generation (GLIGEN).

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        processor ([`~transformers.CLIPProcessor`]):
            A `CLIPProcessor` to procces reference image.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen image-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        image_project ([`CLIPImageProjection`]):
            A `CLIPImageProjection` to project image embedding into phrases embedding space.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _exclude_from_cpu_offload = ["safety_checker"]


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion_k_diffusion.pipeline_stable_diffusion_k_diffusion.StableDiffusionKDiffusionPipeline.check_inputs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents









    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)