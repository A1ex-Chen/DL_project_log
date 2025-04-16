def get_config(config_path, trained: bool=False):
    """
    Returns a config object for a model in model zoo.

    Args:
        config_path (str): config file name relative to detectron2's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
        trained (bool): If True, will set ``MODEL.WEIGHTS`` to trained model zoo weights.
            If False, the checkpoint specified in the config file's ``MODEL.WEIGHTS`` is used
            instead; this will typically (though not always) initialize a subset of weights using
            an ImageNet pre-trained model, while randomly initializing the other weights.

    Returns:
        CfgNode or omegaconf.DictConfig: a config object
    """
    cfg_file = get_config_file(config_path)
    if cfg_file.endswith('.yaml'):
        cfg = get_cfg()
        cfg.merge_from_file(cfg_file)
        if trained:
            cfg.MODEL.WEIGHTS = get_checkpoint_url(config_path)
        return cfg
    elif cfg_file.endswith('.py'):
        cfg = LazyConfig.load(cfg_file)
        if trained:
            url = get_checkpoint_url(config_path)
            if 'train' in cfg and 'init_checkpoint' in cfg.train:
                cfg.train.init_checkpoint = url
            else:
                raise NotImplementedError
        return cfg
