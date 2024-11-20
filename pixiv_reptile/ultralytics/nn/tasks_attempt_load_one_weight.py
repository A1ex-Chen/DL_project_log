def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, weight = torch_safe_load(weight)
    args = {**DEFAULT_CFG_DICT, **ckpt.get('train_args', {})}
    model = (ckpt.get('ema') or ckpt['model']).to(device).float()
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}
    model.pt_path = weight
    model.task = guess_model_task(model)
    if not hasattr(model, 'stride'):
        model.stride = torch.tensor([32.0])
    model = model.fuse().eval() if fuse and hasattr(model, 'fuse'
        ) else model.eval()
    for m in model.modules():
        if hasattr(m, 'inplace'):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m,
            'recompute_scale_factor'):
            m.recompute_scale_factor = None
    return model, ckpt
