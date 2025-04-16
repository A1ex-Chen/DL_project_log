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

import types

import torch
import torch.nn as nn

from deeplite_torch_zoo.utils import get_layerwise_metric_values, reshape_elements
from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES
from deeplite_torch_zoo.src.zero_cost_proxies.utils import aggregate_statistic






@ZERO_COST_SCORES.register('fisher')

                return hook

            layer.dummy.register_backward_hook(hook_factory(layer))

    inputs, outputs, targets, loss_kwargs = next(model_output_generator(model))
    loss = loss_fn(outputs, targets, **loss_kwargs)
    loss.backward()

    # Retrieve fisher info
    def fisher(module):
        if module.fisher is not None:
            return torch.abs(module.fisher.detach())
        else:
            return torch.zeros(module.weight.shape[0])  # size=ch

    grads_abs_ch = get_layerwise_metric_values(model, fisher,
                                               target_layer_types=(nn.Conv2d, nn.Linear))
    shapes = get_layerwise_metric_values(model, lambda l: l.weight.shape[1:])
    grads_abs = reshape_elements(grads_abs_ch, shapes, inputs.device)
    return aggregate_statistic(grads_abs, reduction=reduction)