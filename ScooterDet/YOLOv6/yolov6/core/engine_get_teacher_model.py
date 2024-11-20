def get_teacher_model(self, args, cfg, nc, device):
    teacher_fuse_ab = False if cfg.model.head.num_layers != 3 else True
    model = build_model(cfg, nc, device, fuse_ab=teacher_fuse_ab)
    weights = args.teacher_model_path
    if weights:
        LOGGER.info(f'Loading state_dict from {weights} for teacher')
        model = load_state_dict(weights, model, map_location=device)
    LOGGER.info('Model: {}'.format(model))
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats = False
    return model
