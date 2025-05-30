# Inspired by: https://github.com/haofanwang/ControlNet-for-Diffusers/

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel, logging
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
        >>> from stable_diffusion_controlnet_inpaint_img2img import StableDiffusionControlNetInpaintImg2ImgPipeline

        >>> from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
        >>> from diffusers import ControlNetModel, UniPCMultistepScheduler
        >>> from diffusers.utils import load_image

        >>> def ade_palette():
                return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                        [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                        [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                        [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                        [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                        [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                        [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                        [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                        [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                        [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                        [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                        [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                        [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                        [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                        [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                        [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                        [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                        [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                        [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                        [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                        [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                        [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                        [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                        [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                        [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                        [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                        [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                        [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                        [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                        [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                        [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                        [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                        [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                        [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                        [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                        [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                        [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                        [102, 255, 0], [92, 0, 255]]

        >>> image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        >>> image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

        >>> controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16)

        >>> pipe = StableDiffusionControlNetInpaintImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
            )

        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        >>> pipe.enable_xformers_memory_efficient_attention()
        >>> pipe.enable_model_cpu_offload()

        >>> def image_to_seg(image):
                pixel_values = image_processor(image, return_tensors="pt").pixel_values
                with torch.no_grad():
                    outputs = image_segmentor(pixel_values)
                seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
                color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
                palette = np.array(ade_palette())
                for label, color in enumerate(palette):
                    color_seg[seg == label, :] = color
                color_seg = color_seg.astype(np.uint8)
                seg_image = Image.fromarray(color_seg)
                return seg_image

        >>> image = load_image(
                "https://github.com/CompVis/latent-diffusion/raw/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
            )

        >>> mask_image = load_image(
                "https://github.com/CompVis/latent-diffusion/raw/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
            )

        >>> controlnet_conditioning_image = image_to_seg(image)

        >>> image = pipe(
                "Face of a yellow cat, high resolution, sitting on a park bench",
                image,
                mask_image,
                controlnet_conditioning_image,
                num_inference_steps=20,
            ).images[0]

        >>> image.save("out.png")
        ```
"""








class StableDiffusionControlNetInpaintImg2ImgPipeline(DiffusionPipeline, StableDiffusionMixin):
    """
    Inspired by: https://github.com/haofanwang/ControlNet-for-Diffusers/
    """

    _optional_components = ["safety_checker", "feature_extractor"]












    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)