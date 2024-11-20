def get_optimizer_param(args, cfg, model):
    """ Build optimizer from cfg file."""
    accumulate = max(1, round(64 / args.batch_size))
    cfg.solver.weight_decay *= args.batch_size * accumulate / 64
    g_bnw, g_w, g_b = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g_b.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g_bnw.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g_w.append(v.weight)
    return [{'params': g_bnw}, {'params': g_w, 'weight_decay': cfg.solver.
        weight_decay}, {'params': g_b}]
