def __init__(self, model, scales, args, cfg, momentum=0, dampening=0,
    weight_decay=0, nesterov=True, reinit=True,
    use_identity_scales_for_reinit=True, cpu_mode=False):
    defaults = dict(lr=cfg.solver.lr0, momentum=cfg.solver.momentum,
        dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
    if nesterov and (cfg.solver.momentum <= 0 or dampening != 0):
        raise ValueError(
            'Nesterov momentum requires a momentum and zero dampening')
    parameters = get_optimizer_param(args, cfg, model)
    super(SGD, self).__init__(parameters, defaults)
    self.num_layers = len(scales)
    blocks = []
    extract_blocks_into_list(model, blocks)
    convs = [b.conv for b in blocks]
    assert len(scales) == len(convs)
    if reinit:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                gamma_init = m.weight.mean()
                if gamma_init == 1.0:
                    LOGGER.info('Checked. This is training from scratch.')
                else:
                    LOGGER.warning(
                        '========================== Warning! Is this really training from scratch ? ================='
                        )
        LOGGER.info('##################### Re-initialize #############')
        self.reinitialize(scales, convs, use_identity_scales_for_reinit)
    self.generate_gradient_masks(scales, convs, cpu_mode)
