def attempt_load(weights, device=None, inplace=True, fuse=True):
    from models.yolo import Detect, Model
    model = Ensemble()
    for w in (weights if isinstance(weights, list) else [weights]):
        ckpt = torch.load(attempt_download(w), map_location='cpu')
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()
        model.append(ckpt.fuse().eval() if fuse else ckpt.eval())
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU,
            Detect, Model):
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None
    if len(model) == 1:
        return model[-1]
    print(f'Ensemble created with {weights}\n')
    for k in ('names', 'nc', 'yaml'):
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in
        model])).int()].stride
    assert all(model[0].nc == m.nc for m in model
        ), f'Models have different class counts: {[m.nc for m in model]}'
    return model