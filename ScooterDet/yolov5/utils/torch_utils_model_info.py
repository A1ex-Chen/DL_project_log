def model_info(model, verbose=False, imgsz=640):
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    if verbose:
        print(
            f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}"
            )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' % (i, name, p.
                requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    try:
        p = next(model.parameters())
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride'
            ) else 32
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)
        flops = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0
            ] / 1000000000.0 * 2
        imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]
        fs = f', {flops * imgsz[0] / stride * imgsz[1] / stride:.1f} GFLOPs'
    except Exception:
        fs = ''
    name = Path(model.yaml_file).stem.replace('yolov5', 'YOLOv5') if hasattr(
        model, 'yaml_file') else 'Model'
    LOGGER.info(
        f'{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}'
        )
