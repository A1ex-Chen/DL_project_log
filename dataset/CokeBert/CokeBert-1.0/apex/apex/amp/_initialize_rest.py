import torch
from torch._six import string_classes
import functools
import numpy as np
import warnings
from ._amp_state import _amp_state, warn_or_err, container_abcs
from .handle import disable_casts
from .scaler import LossScaler
from ._process_optimizer import _process_optimizer
from apex.fp16_utils import convert_network
from ..fp16_utils import FP16_Optimizer as FP16_Optimizer_general
from ..optimizers import FP16_Optimizer as FP16_Optimizer_for_fused
from ..optimizers import FusedAdam
from ..parallel import DistributedDataParallel as apex_DDP
from ..parallel.LARC import LARC




# Modified from torch.optim.optimizer.py.  This is a bit more general than casted_args in utils.py.











            model.forward = patch_forward(model.forward)

        # State dict trick to recast any preexisting per-param state tensors 
        for optimizer in optimizers:
            optimizer.load_state_dict(optimizer.state_dict())
    elif cast_model_outputs is not None:
        output_caster = functools.partial(to_type, cast_model_outputs)

        for model in models:
            def patch_forward(old_fwd):
                def new_fwd(*args, **kwargs):
                    output = old_fwd(*args, **kwargs)
                    return applier(output, output_caster)
                return new_fwd

            model.forward = patch_forward(model.forward)

    for i, optimizer in enumerate(optimizers):
        # Still need to special case this for the first pass
        if isinstance(optimizer, FusedAdam):
            optimizers[i] = wrap_fused_adam(optimizer, properties)
        else:
            optimizers[i] = _process_optimizer(optimizer, properties)

    _amp_state.loss_scalers = []
    for _ in range(num_losses):
        _amp_state.loss_scalers.append(LossScaler(properties.loss_scale,
                                                  min_loss_scale=_amp_state.min_loss_scale,
                                                  max_loss_scale=_amp_state.max_loss_scale))

    if properties.patch_torch_functions:
        # handle is unused here. It's accessible later through a global value anyway.
        handle = amp_init(loss_scale=properties.loss_scale, verbose=(_amp_state.verbosity == 2))
        for optimizer in optimizers:
            # Disable Amp casting for the optimizer step, because it should only be
            # applied to FP32 master params anyway.
                return new_fwd

            model.forward = patch_forward(model.forward)

        # State dict trick to recast any preexisting per-param state tensors 
        for optimizer in optimizers:
            optimizer.load_state_dict(optimizer.state_dict())
    elif cast_model_outputs is not None:
        output_caster = functools.partial(to_type, cast_model_outputs)

        for model in models:

            model.forward = patch_forward(model.forward)

    for i, optimizer in enumerate(optimizers):
        # Still need to special case this for the first pass
        if isinstance(optimizer, FusedAdam):
            optimizers[i] = wrap_fused_adam(optimizer, properties)
        else:
            optimizers[i] = _process_optimizer(optimizer, properties)

    _amp_state.loss_scalers = []
    for _ in range(num_losses):
        _amp_state.loss_scalers.append(LossScaler(properties.loss_scale,
                                                  min_loss_scale=_amp_state.min_loss_scale,
                                                  max_loss_scale=_amp_state.max_loss_scale))

    if properties.patch_torch_functions:
        # handle is unused here. It's accessible later through a global value anyway.
        handle = amp_init(loss_scale=properties.loss_scale, verbose=(_amp_state.verbosity == 2))
        for optimizer in optimizers:
            # Disable Amp casting for the optimizer step, because it should only be
            # applied to FP32 master params anyway.
            def patch_step(old_step):
                return new_fwd

            model.forward = patch_forward(model.forward)

    for i, optimizer in enumerate(optimizers):
        # Still need to special case this for the first pass
        if isinstance(optimizer, FusedAdam):
            optimizers[i] = wrap_fused_adam(optimizer, properties)
        else:
            optimizers[i] = _process_optimizer(optimizer, properties)

    _amp_state.loss_scalers = []
    for _ in range(num_losses):
        _amp_state.loss_scalers.append(LossScaler(properties.loss_scale,
                                                  min_loss_scale=_amp_state.min_loss_scale,
                                                  max_loss_scale=_amp_state.max_loss_scale))

    if properties.patch_torch_functions:
        # handle is unused here. It's accessible later through a global value anyway.
        handle = amp_init(loss_scale=properties.loss_scale, verbose=(_amp_state.verbosity == 2))
        for optimizer in optimizers:
            # Disable Amp casting for the optimizer step, because it should only be
            # applied to FP32 master params anyway.
            def patch_step(old_step):
                def new_step(*args, **kwargs):
                    with disable_casts():
                        output = old_step(*args, **kwargs)
                    return output
                return new_step

            optimizer.step = patch_step(optimizer.step)

    if optimizers_was_list:
        if models_was_list:
            return models, optimizers
        else:
            return models[0], optimizers
    else:
        if models_was_list:
            if len(optimizers) == 0:
                return models
            else:
                return models, optimizers[0]
        else:
            if len(optimizers) == 0:
                return models[0]
            else:
                return models[0], optimizers[0]