# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import gc
import unittest

from diffusers import FlaxStableDiffusionInpaintPipeline
from diffusers.utils import is_flax_available, load_image
from diffusers.utils.testing_utils import require_flax, slow


if is_flax_available():
    import jax
    import jax.numpy as jnp
    from flax.jax_utils import replicate
    from flax.training.common_utils import shard


@slow
@require_flax
class FlaxStableDiffusionInpaintPipelineIntegrationTests(unittest.TestCase):
