# Copyright 2023 Kakao Brain and The HuggingFace Team. All rights reserved.
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
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin


class UnCLIPTextProjModel(ModelMixin, ConfigMixin):
    """
    Utility class for CLIP embeddings. Used to combine the image and text embeddings into a format usable by the
    decoder.

    For more details, see the original paper: https://arxiv.org/abs/2204.06125 section 2.1
    """

    @register_to_config
