# Copyright 2023 Microsoft and The HuggingFace Team. All rights reserved.
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

from typing import Callable, List, Optional, Tuple, Union

import torch
from transformers import CLIPTextModel, CLIPTokenizer

from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin, Transformer2DModel, VQModel
from ...schedulers import VQDiffusionScheduler
from ...utils import logging
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LearnedClassifierFreeSamplingEmbeddings(ModelMixin, ConfigMixin):
    """
    Utility class for storing learned text embeddings for classifier free sampling
    """

    @register_to_config


class VQDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using VQ Diffusion

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vqvae ([`VQModel`]):
            Vector Quantized Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent
            representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. VQ Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        transformer ([`Transformer2DModel`]):
            Conditional transformer to denoise the encoded image latents.
        scheduler ([`VQDiffusionScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    vqvae: VQModel
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    transformer: Transformer2DModel
    learned_classifier_free_sampling_embeddings: LearnedClassifierFreeSamplingEmbeddings
    scheduler: VQDiffusionScheduler



    @torch.no_grad()
