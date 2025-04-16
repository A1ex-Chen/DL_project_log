def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay
    =1e-05, iterations=100000.0):
    """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
    g = [], [], []
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
    if name == 'auto':
        LOGGER.info(
            f"{colorstr('optimizer:')} 'optimizer=auto' found, ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
        nc = getattr(model, 'nc', 10)
        lr_fit = round(0.002 * 5 / (4 + nc), 6)
        name, lr, momentum = ('SGD', 0.01, 0.9) if iterations > 10000 else (
            'AdamW', lr_fit, 0.9)
        self.args.warmup_bias_lr = 0.0
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = (f'{module_name}.{param_name}' if module_name else
                param_name)
            if 'bias' in fullname:
                g[2].append(param)
            elif isinstance(module, bn):
                g[1].append(param)
            else:
                g[0].append(param)
    if name in {'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam'}:
        optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(
            momentum, 0.999), weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(
            f"Optimizer '{name}' not found in list of available optimizers [Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto].To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics."
            )
    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})
    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups {len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
    return optimizer
