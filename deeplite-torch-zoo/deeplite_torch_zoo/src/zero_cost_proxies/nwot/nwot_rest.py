# Credit: https://github.com/BayesWatch/nas-without-training

from functools import partial

import torch
import numpy as np

from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES
from deeplite_torch_zoo.src.dnn_blocks.common import ACT_TYPE_MAP
from deeplite_torch_zoo.utils import TRAINABLE_LAYERS


ACTIVATION_TYPES = tuple(type(module) for module in ACT_TYPE_MAP.values())




@ZERO_COST_SCORES.register('nwot')

    model.K = 0. if reduction == 'sum' else []
    for module in model.modules():
        if (pre_act and isinstance(module, TRAINABLE_LAYERS)) or (not pre_act and isinstance(module, ACTIVATION_TYPES)):
            module.register_forward_hook(counting_forward_hook)

    with torch.no_grad():
        model(inputs)

    return logdet(model.K) if reduction == 'sum' else model.K


ZERO_COST_SCORES.register('nwot_preact')(partial(compute_nwot, pre_act=True))