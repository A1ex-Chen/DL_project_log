def model_info(model, verbose=False, img_size=640):
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name',
            'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' % (i, name, p.
                requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    try:
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride'
            ) else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride),
            device=next(model.parameters()).device)
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0
            ] / 1000000000.0 * 2
        img_size = img_size if isinstance(img_size, list) else [img_size,
            img_size]
        fs = ', %.1f GFLOPS' % (flops * img_size[0] / stride * img_size[1] /
            stride)
    except (ImportError, Exception):
        fs = ''
    logger.info(
        f'Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}'
        )
