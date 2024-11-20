def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""
    ensemble = Ensemble()
    for w in (weights if isinstance(weights, list) else [weights]):
        ckpt, w = torch_safe_load(w)
        args = {**DEFAULT_CFG_DICT, **ckpt['train_args']
            } if 'train_args' in ckpt else None
        model = (ckpt.get('ema') or ckpt['model']).to(device).float()
        model.args = args
        model.pt_path = w
        model.task = guess_model_task(model)
        if not hasattr(model, 'stride'):
            model.stride = torch.tensor([32.0])
        ensemble.append(model.fuse().eval() if fuse and hasattr(model,
            'fuse') else model.eval())
    for m in ensemble.modules():
        if hasattr(m, 'inplace'):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m,
            'recompute_scale_factor'):
            m.recompute_scale_factor = None
    if len(ensemble) == 1:
        return ensemble[-1]
    LOGGER.info(f'Ensemble created with {weights}\n')
    for k in ('names', 'nc', 'yaml'):
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max(
        ) for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble
        ), f'Models differ in class counts {[m.nc for m in ensemble]}'
    return ensemble
