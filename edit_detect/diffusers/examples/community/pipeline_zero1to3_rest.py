# A diffuser version implementation of Zero1to3 (https://github.com/cvlab-columbia/zero123), ICCV 2023
# by Xin Kong

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import kornia
import numpy as np
import PIL.Image
import torch
from packaging import version
from transformers import CLIPFeatureExtractor, CLIPVisionModelWithProjection

# from ...configuration_utils import FrozenDict
# from ...models import AutoencoderKL, UNet2DConditionModel
# from ...schedulers import KarrasDiffusionSchedulers
# from ...utils import (
#     deprecate,
#     is_accelerate_available,
#     is_accelerate_version,
#     logging,
#     randn_tensor,
#     replace_example_docstring,
# )
# from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# from . import StableDiffusionPipelineOutput
# from .safety_checker import StableDiffusionSafetyChecker
from diffusers import AutoencoderKL, DiffusionPipeline, StableDiffusionMixin, UNet2DConditionModel
from diffusers.configuration_utils import ConfigMixin, FrozenDict
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
# todo
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


class CCProjection(ModelMixin, ConfigMixin):



class Zero1to3StableDiffusionPipeline(DiffusionPipeline, StableDiffusionMixin):
    r"""
    Pipeline for single view conditioned novel view generation using Zero1to3.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder. Stable Diffusion Image Variation uses the vision portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
        cc_projection ([`CCProjection`]):
            Projection layer to project the concated CLIP features and pose embeddings to the original CLIP feature size.
    """

    _optional_components = ["safety_checker", "feature_extractor"]

        # self.model_mode = None



    # from image_variation









    # def load_cc_projection(self, pretrained_weights=None):
    #     self.cc_projection = torch.nn.Linear(772, 768)
    #     torch.nn.init.eye_(list(self.cc_projection.parameters())[0][:768, :768])
    #     torch.nn.init.zeros_(list(self.cc_projection.parameters())[1])
    #     if pretrained_weights is not None:
    #         self.cc_projection.load_state_dict(pretrained_weights)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)