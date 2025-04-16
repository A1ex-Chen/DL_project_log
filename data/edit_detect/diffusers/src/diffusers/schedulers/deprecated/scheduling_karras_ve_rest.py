# Copyright 2024 NVIDIA and The HuggingFace Team. All rights reserved.
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


from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput
from ...utils.torch_utils import randn_tensor
from ..scheduling_utils import SchedulerMixin


@dataclass
class KarrasVeOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        derivative (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Derivative of predicted original image sample (x_0).
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    derivative: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class KarrasVeScheduler(SchedulerMixin, ConfigMixin):
    """
    A stochastic scheduler tailored to variance-expanding models.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    <Tip>

    For more details on the parameters, see [Appendix E](https://arxiv.org/abs/2206.00364). The grid search values used
    to find the optimal `{s_noise, s_churn, s_min, s_max}` for a specific model are described in Table 5 of the paper.

    </Tip>

    Args:
        sigma_min (`float`, defaults to 0.02):
            The minimum noise magnitude.
        sigma_max (`float`, defaults to 100):
            The maximum noise magnitude.
        s_noise (`float`, defaults to 1.007):
            The amount of additional noise to counteract loss of detail during sampling. A reasonable range is [1.000,
            1.011].
        s_churn (`float`, defaults to 80):
            The parameter controlling the overall amount of stochasticity. A reasonable range is [0, 100].
        s_min (`float`, defaults to 0.05):
            The start value of the sigma range to add noise (enable stochasticity). A reasonable range is [0, 10].
        s_max (`float`, defaults to 50):
            The end value of the sigma range to add noise. A reasonable range is [0.2, 80].
    """

    order = 2

    @register_to_config





