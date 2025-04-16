def get_model(self, args, cfg, nc, device):
    if 'YOLOv6-lite' in cfg.model.type:
        assert not self.args.fuse_ab, 'ERROR in: YOLOv6-lite models not support fuse_ab mode.'
        assert not self.args.distill, 'ERROR in: YOLOv6-lite models not support distill mode.'
        model = build_lite_model(cfg, nc, device)
    else:
        model = build_model(cfg, nc, device, fuse_ab=self.args.fuse_ab,
            distill_ns=self.distill_ns)
    weights = cfg.model.pretrained
    if weights:
        if not os.path.exists(weights):
            download_ckpt(weights)
        LOGGER.info(f'Loading state_dict from {weights} for fine-tuning...')
        model = load_state_dict(weights, model, map_location=device)
    LOGGER.info('Model: {}'.format(model))
    return model
