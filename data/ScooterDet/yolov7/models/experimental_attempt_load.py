def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in (weights if isinstance(weights, list) else [weights]):
        attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().
            fuse().eval())
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()
    if len(model) == 1:
        return model[-1]
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model
