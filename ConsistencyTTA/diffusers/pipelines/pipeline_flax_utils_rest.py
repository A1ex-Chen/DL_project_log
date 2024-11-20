# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import importlib
import inspect
import os
from typing import Any, Dict, List, Optional, Union

import flax
import numpy as np
import PIL
from flax.core.frozen_dict import FrozenDict
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm.auto import tqdm

from ..configuration_utils import ConfigMixin
from ..models.modeling_flax_utils import FLAX_WEIGHTS_NAME, FlaxModelMixin
from ..schedulers.scheduling_utils_flax import SCHEDULER_CONFIG_NAME, FlaxSchedulerMixin
from ..utils import CONFIG_NAME, DIFFUSERS_CACHE, BaseOutput, http_user_agent, is_transformers_available, logging


if is_transformers_available():
    from transformers import FlaxPreTrainedModel

INDEX_FILE = "diffusion_flax_model.bin"


logger = logging.get_logger(__name__)


LOADABLE_CLASSES = {
    "diffusers": {
        "FlaxModelMixin": ["save_pretrained", "from_pretrained"],
        "FlaxSchedulerMixin": ["save_pretrained", "from_pretrained"],
        "FlaxDiffusionPipeline": ["save_pretrained", "from_pretrained"],
    },
    "transformers": {
        "PreTrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PreTrainedTokenizerFast": ["save_pretrained", "from_pretrained"],
        "FlaxPreTrainedModel": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
        "ProcessorMixin": ["save_pretrained", "from_pretrained"],
        "ImageProcessingMixin": ["save_pretrained", "from_pretrained"],
    },
}

ALL_IMPORTABLE_CLASSES = {}
for library in LOADABLE_CLASSES:
    ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])




@flax.struct.dataclass
class FlaxImagePipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


class FlaxDiffusionPipeline(ConfigMixin):
    r"""
    Base class for all models.

    [`FlaxDiffusionPipeline`] takes care of storing all components (models, schedulers, processors) for diffusion
    pipelines and handles methods for loading, downloading and saving models as well as a few methods common to all
    pipelines to:

        - enabling/disabling the progress bar for the denoising iteration

    Class attributes:

        - **config_name** ([`str`]) -- name of the config file that will store the class and module names of all
          components of the diffusion pipeline.
    """
    config_name = "model_index.json"



    @classmethod

    @staticmethod

    @property

    @staticmethod

    # TODO: make it compatible with jax.lax
