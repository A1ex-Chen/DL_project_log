import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class FP16_Optimizer(object):
    """
    :class:`FP16_Optimizer` A cutdown version of apex.fp16_utils.FP16_Optimizer.
    Designed only to wrap apex.optimizers.FusedAdam.
    Refer to apex.fp16_utils documents for more information.

    Example::

        model = torch.nn.Linear(D_in, D_out).cuda().half()
        optimizer = apex.optimizers.FusedAdam(model.parameters())
        # Name the FP16_Optimizer instance to replace the existing optimizer
        # (recommended but not required):
        optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
        ...
        # loss.backward() becomes:
        optimizer.backward(loss)
        ...

    Example with dynamic loss scaling::

        ...
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
                                   # optional arg to control dynamic loss scaling behavior
                                   # dynamic_loss_args={'scale_window' : 500})
                                   # Usually, dynamic_loss_args is not necessary.
    """







    # Promote state so it can be retrieved or set via "fp16_optimizer_instance.state"


    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via "fp16_optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)


    param_groups = property(_get_param_groups, _set_param_groups)

