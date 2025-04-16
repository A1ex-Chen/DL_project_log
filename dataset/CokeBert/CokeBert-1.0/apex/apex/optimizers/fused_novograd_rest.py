import torch
from apex.multi_tensor_apply import multi_tensor_applier

class FusedNovoGrad(torch.optim.Optimizer):

    """Implements NovoGrad algorithm.

    Currently GPU-only.  Requires Apex to be installed via
    ``pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./``.

    This version of fused NovoGrad implements 2 fusions.

      * Fusion of the NovoGrad update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.

    :class:`apex.optimizers.FusedNovoGrad`'s usage is identical to any Pytorch optimizer::

        opt = apex.optimizers.FusedNovoGrad(model.parameters(), lr = ....)
        ...
        opt.step()

    :class:`apex.optimizers.FusedNovoGrad` may be used with or without Amp.  If you wish to use :class:`FusedNovoGrad` with Amp,
    you may choose any ``opt_level``::

        opt = apex.optimizers.FusedNovoGrad(model.parameters(), lr = ....)
        model, opt = amp.initialize(model, opt, opt_level="O0" or "O1 or "O2")
        ...
        opt.step()

    In general, ``opt_level="O1"`` is recommended.

    It has been proposed in `Jasper: An End-to-End Convolutional Neural Acoustic Model`_.
    More info: https://nvidia.github.io/OpenSeq2Seq/html/optimizers.html#novograd

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            NOT SUPPORTED now! (default: False)
        reg_inside_moment (bool, optional): whether do regularization (norm and L2)
            in momentum calculation. True for include, False for not include and
            only do it on update term. (default: False)
        grad_averaging (bool, optional): whether apply (1-beta2) to grad when
            calculating running averages of gradient. (default: True)
        norm_type (int, optional): which norm to calculate for each layer.
            2 for L2 norm, and 0 for infinite norm. These 2 are only supported
            type now. (default: 2)
        init_zero (bool, optional): whether init norm with 0 (start averaging on
            1st step) or first step norm (start averaging on 2nd step). True for
            init with 0. (default: False)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)

    .. _Jasper\: An End-to-End Convolutional Neural Acoustic Model:
        https://arxiv.org/abs/1904.03288
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """



