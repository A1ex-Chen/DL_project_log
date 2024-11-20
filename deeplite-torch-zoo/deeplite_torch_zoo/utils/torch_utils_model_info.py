def model_info(model, detailed=False, verbose=True, imgsz=640):
    """Model information. imgsz may be int or list, i.e. imgsz=640 or imgsz=[640, 320]."""
    if not verbose:
        return None
    n_p = get_num_params(model)
    n_g = get_num_gradients(model)
    n_l = len(list(model.modules()))
    if detailed:
        LOGGER.info(
            f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}"
            )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            LOGGER.info('%5g %40s %9s %12g %20s %10.3g %10.3g %10s', i,
                name, p.requires_grad, p.numel(), list(p.shape), p.mean(),
                p.std(), p.dtype)
    fused = ' (fused)' if getattr(model, 'is_fused', lambda : False)() else ''
    model_name = 'Model'
    if hasattr(model, 'yaml') and model.yaml is not None or hasattr(model,
        'yaml_file'):
        yaml_file = getattr(model, 'yaml_file', '') or getattr(model,
            'yaml', {}).get('yaml_file', '')
        model_name = Path(yaml_file).stem.replace('yolo', 'YOLO') or 'Model'
    LOGGER.info(
        f'{model_name} summary{fused}: {n_l} layers, {n_p} parameters, {n_g} gradients'
        )
    return n_l, n_p, n_g
