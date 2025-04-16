import types
from typing import List, Optional, Tuple, Union

import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

from diffusers.models import PriorTransformer
from diffusers.pipelines import DiffusionPipeline, StableDiffusionImageVariationPipeline
from diffusers.schedulers import UnCLIPScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name




class StableUnCLIPPipeline(DiffusionPipeline):


    @property



    @torch.no_grad()