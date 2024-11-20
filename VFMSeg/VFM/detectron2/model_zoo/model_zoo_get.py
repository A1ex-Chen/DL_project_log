def get(config_path, trained: bool=False, device: Optional[str]=None):
    """
    Get a model specified by relative path under Detectron2's official ``configs/`` directory.

    Args:
        config_path (str): config file name relative to detectron2's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
        trained (bool): see :func:`get_config`.
        device (str or None): overwrite the device in config, if given.

    Returns:
        nn.Module: a detectron2 model. Will be in training mode.

    Example:
    ::
        from detectron2 import model_zoo
        model = model_zoo.get("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml", trained=True)
    """
    cfg = get_config(config_path, trained)
    if device is None and not torch.cuda.is_available():
        device = 'cpu'
    if device is not None and isinstance(cfg, CfgNode):
        cfg.MODEL.DEVICE = device
    if isinstance(cfg, CfgNode):
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        model = instantiate(cfg.model)
        if device is not None:
            model = model.to(device)
        if 'train' in cfg and 'init_checkpoint' in cfg.train:
            DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    return model
