from typing import List, Optional, Tuple, Union

import torch

from diffusers import DiffusionPipeline
from diffusers.configuration_utils import ConfigMixin
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin


class IADBScheduler(SchedulerMixin, ConfigMixin):
    """
    IADBScheduler is a scheduler for the Iterative α-(de)Blending denoising method. It is simple and minimalist.

    For more details, see the original paper: https://arxiv.org/abs/2305.03486 and the blog post: https://ggx-research.github.io/publication/2023/05/10/publication-iadb.html
    """






class IADBPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """


    @torch.no_grad()