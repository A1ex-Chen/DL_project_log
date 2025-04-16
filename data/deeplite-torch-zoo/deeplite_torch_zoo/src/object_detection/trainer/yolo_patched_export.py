def patched_export(obj, model_name='model', **kwargs):
    obj.model.yaml = {'yaml_file': model_name}
    obj._check_is_pytorch_model()
    overrides = obj.overrides.copy()
    overrides.update(kwargs)
    overrides['mode'] = 'export'
    args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
    args.task = obj.task
    if args.imgsz == DEFAULT_CFG.imgsz:
        args.imgsz = obj.model.args['imgsz']
    if args.batch == DEFAULT_CFG.batch:
        args.batch = 1
    model = deepcopy(obj.model).to(obj.device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()
    for k, m in model.named_modules():
        if isinstance(m, tuple(HEAD_NAME_MAP.values())):
            m.dynamic = args.dynamic
            m.export = True
            m.format = args.format
    model.yaml_file = model_name
    return Exporter(overrides=args, _callbacks=obj.callbacks)(model=model)
