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

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from transformers import CLIPConfig, FlaxPreTrainedModel
from transformers.models.clip.modeling_flax_clip import FlaxCLIPVisionModule




class FlaxStableDiffusionSafetyCheckerModule(nn.Module):
    config: CLIPConfig
    dtype: jnp.dtype = jnp.float32




class FlaxStableDiffusionSafetyChecker(FlaxPreTrainedModel):
    config_class = CLIPConfig
    main_input_name = "clip_input"
    module_class = FlaxStableDiffusionSafetyCheckerModule


