# Copyright 2022 The Music Spectrogram Diffusion Authors.
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import torch
import torch.nn as nn
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.t5.modeling_t5 import T5Block, T5Config, T5LayerNorm

from ....configuration_utils import ConfigMixin, register_to_config
from ....models import ModelMixin


class SpectrogramNotesEncoder(ModelMixin, ConfigMixin, ModuleUtilsMixin):
    @register_to_config
