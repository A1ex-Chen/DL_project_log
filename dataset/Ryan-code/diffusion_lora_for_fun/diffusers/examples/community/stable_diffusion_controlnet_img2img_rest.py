# Inspired by: https://github.com/haofanwang/ControlNet-for-Diffusers/

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel, logging
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    PIL_INTERPOLATION,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import numpy as np
        >>> import torch
        >>> from PIL import Image
        >>> from diffusers import ControlNetModel, UniPCMultistepScheduler
        >>> from diffusers.utils import load_image

        >>> input_image = load_image("https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")

        >>> controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

        >>> pipe_controlnet = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                safety_checker=None,
                torch_dtype=torch.float16
                )

        >>> pipe_controlnet.scheduler = UniPCMultistepScheduler.from_config(pipe_controlnet.scheduler.config)
        >>> pipe_controlnet.enable_xformers_memory_efficient_attention()
        >>> pipe_controlnet.enable_model_cpu_offload()

        # using image with edges for our canny controlnet
        >>> control_image = load_image(
            "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/vermeer_canny_edged.png")


        >>> result_img = pipe_controlnet(controlnet_conditioning_image=control_image,
                        image=input_image,
                        prompt="an android robot, cyberpank, digitl art masterpiece",
                        num_inference_steps=20).images[0]

        >>> result_img.show()
        ```
"""






class StableDiffusionControlNetImg2ImgPipeline(DiffusionPipeline, StableDiffusionMixin):
    """
    Inspired by: https://github.com/haofanwang/ControlNet-for-Diffusers/
    """

    _optional_components = ["safety_checker", "feature_extractor"]











    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)