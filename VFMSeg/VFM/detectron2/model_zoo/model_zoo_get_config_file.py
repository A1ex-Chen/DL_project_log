def get_config_file(config_path):
    """
    Returns path to a builtin config file.

    Args:
        config_path (str): config file name relative to detectron2's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

    Returns:
        str: the real path to the config file.
    """
    cfg_file = pkg_resources.resource_filename('detectron2.model_zoo', os.
        path.join('configs', config_path))
    if not os.path.exists(cfg_file):
        raise RuntimeError('{} not available in Model Zoo!'.format(config_path)
            )
    return cfg_file
