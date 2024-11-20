from logging import getLogger
from typing import Any, Callable, List, Optional, Union

import numpy as np
import PIL
import torch

from ...schedulers import DDPMScheduler
from ..onnx_utils import ORT_TO_NP_TYPE, OnnxRuntimeModel
from ..pipeline_utils import ImagePipelineOutput
from . import StableDiffusionUpscalePipeline


logger = getLogger(__name__)


NUM_LATENT_CHANNELS = 4
NUM_UNET_INPUT_CHANNELS = 7

ORT_TO_PT_TYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
}




class OnnxStableDiffusionUpscalePipeline(StableDiffusionUpscalePipeline):


