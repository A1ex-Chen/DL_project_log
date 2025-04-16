def get_model_no_weights(config_path):
    """
    Like model_zoo.get, but do not load any weights (even pretrained)
    """
    cfg = model_zoo.get_config(config_path)
    if isinstance(cfg, CfgNode):
        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE = 'cpu'
        return build_model(cfg)
    else:
        return instantiate(cfg.model)
