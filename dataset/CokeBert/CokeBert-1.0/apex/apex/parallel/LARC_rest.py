import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class LARC(object):
    """
    :class:`LARC` is a pytorch implementation of both the scaling and clipping variants of LARC,
    in which the ratio between gradient and parameter magnitudes is used to calculate an adaptive 
    local learning rate for each individual parameter. The algorithm is designed to improve
    convergence of large batch training.
     
    See https://arxiv.org/abs/1708.03888 for calculation of the local learning rate.

    In practice it modifies the gradients of parameters as a proxy for modifying the learning rate
    of the parameters. This design allows it to be used as a wrapper around any torch.optim Optimizer.

    ```
    model = ...
    optim = torch.optim.Adam(model.parameters(), lr=...)
    optim = LARC(optim)
    ```

    It can even be used in conjunction with apex.fp16_utils.FP16_optimizer.

    ```
    model = ...
    optim = torch.optim.Adam(model.parameters(), lr=...)
    optim = LARC(optim)
    optim = apex.fp16_utils.FP16_Optimizer(optim)
    ```

    Args:
        optimizer: Pytorch optimizer to wrap and modify learning rate for.
        trust_coefficient: Trust coefficient for calculating the lr. See https://arxiv.org/abs/1708.03888
        clip: Decides between clipping or scaling mode of LARC. If `clip=True` the learning rate is set to `min(optimizer_lr, local_lr)` for each parameter. If `clip=False` the learning rate is set to `local_lr*optimizer_lr`.
        eps: epsilon kludge to help with numerical stability while calculating adaptive_lr
    """








