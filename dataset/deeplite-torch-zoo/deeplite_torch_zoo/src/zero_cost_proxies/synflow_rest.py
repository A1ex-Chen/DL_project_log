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

from functools import partial

import torch
import torch.nn as nn

from deeplite_torch_zoo.utils import get_layerwise_metric_values, NORMALIZATION_LAYERS
from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES
from deeplite_torch_zoo.src.zero_cost_proxies.utils import aggregate_statistic






@ZERO_COST_SCORES.register('synflow')

    model.double()

    if not bn_training_mode:
        if not dummify_bns:
            model.eval()
        linearize(model)

    inp, _, _, _ = next(model_output_generator(nn.Identity()))
    # compute gradients with input of all-ones
    inputs = torch.ones((1, *inp.shape[1:]), device=inp.device, dtype=torch.float64)
    outputs = model(inputs)
    torch.sum(output_post_processing(outputs)).backward()

    # select the gradients
    grads_abs = get_layerwise_metric_values(model, get_synflow)
    return aggregate_statistic(grads_abs, reduction=reduction)


synflow_train = partial(synflow, dummify_bns=False, bn_training_mode=True)
synflow_train = ZERO_COST_SCORES.register('synflow_train')(synflow_train)