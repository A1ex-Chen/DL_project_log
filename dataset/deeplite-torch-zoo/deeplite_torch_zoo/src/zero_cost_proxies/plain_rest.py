# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch

from deeplite_torch_zoo.utils import get_layerwise_metric_values
from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES
from deeplite_torch_zoo.src.zero_cost_proxies.utils import aggregate_statistic


@ZERO_COST_SCORES.register('plain')

    grads_abs = get_layerwise_metric_values(model, plain)
    return aggregate_statistic(grads_abs, reduction=reduction)