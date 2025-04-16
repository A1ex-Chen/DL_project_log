# coding=utf-8
# Copyright 2018 The Google Flax Team Authors and The HuggingFace Inc. team.
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

import os
from abc import ABC, abstractmethod
from pickle import UnpicklingError
from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.serialization import to_bytes
from flax.traverse_util import unflatten_dict
from jax.random import PRNGKey

from .configuration_utils import PretrainedConfig
from .file_utils import WEIGHTS_NAME, cached_path, hf_bucket_url, is_remote_url
from .utils import logging


logger = logging.get_logger(__name__)


@jax.jit


ACT2FN = {
    "gelu": nn.gelu,
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": gelu,
}


class FlaxPreTrainedModel(ABC):
    config_class = None
    pretrained_model_archive_map = {}
    base_model_prefix = ""
    model_class = None


    @property

    @staticmethod
    @abstractmethod

    @classmethod
