# Copyright 2024 Kakao Brain and The HuggingFace Team. All rights reserved.
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
from typing import List, Optional, Tuple, Union

import torch
from torch.nn import functional as F
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

from ...models import PriorTransformer, UNet2DConditionModel, UNet2DModel
from ...schedulers import UnCLIPScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from .text_proj import UnCLIPTextProjModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class UnCLIPPipeline(DiffusionPipeline):
    """
    Pipeline for text-to-image generation using unCLIP.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        text_encoder ([`~transformers.CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        prior ([`PriorTransformer`]):
            The canonical unCLIP prior to approximate the image embedding from the text embedding.
        text_proj ([`UnCLIPTextProjModel`]):
            Utility class to prepare and combine the embeddings before they are passed to the decoder.
        decoder ([`UNet2DConditionModel`]):
            The decoder to invert the image embedding into an image.
        super_res_first ([`UNet2DModel`]):
            Super resolution UNet. Used in all but the last step of the super resolution diffusion process.
        super_res_last ([`UNet2DModel`]):
            Super resolution UNet. Used in the last step of the super resolution diffusion process.
        prior_scheduler ([`UnCLIPScheduler`]):
            Scheduler used in the prior denoising process (a modified [`DDPMScheduler`]).
        decoder_scheduler ([`UnCLIPScheduler`]):
            Scheduler used in the decoder denoising process (a modified [`DDPMScheduler`]).
        super_res_scheduler ([`UnCLIPScheduler`]):
            Scheduler used in the super resolution denoising process (a modified [`DDPMScheduler`]).

    """

    _exclude_from_cpu_offload = ["prior"]

    prior: PriorTransformer
    decoder: UNet2DConditionModel
    text_proj: UnCLIPTextProjModel
    text_encoder: CLIPTextModelWithProjection
    tokenizer: CLIPTokenizer
    super_res_first: UNet2DModel
    super_res_last: UNet2DModel

    prior_scheduler: UnCLIPScheduler
    decoder_scheduler: UnCLIPScheduler
    super_res_scheduler: UnCLIPScheduler

    model_cpu_offload_seq = "text_encoder->text_proj->decoder->super_res_first->super_res_last"




    @torch.no_grad()