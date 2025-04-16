def get_checkpoint_url(config_path):
    """
    Returns the URL to the model trained using the given config

    Args:
        config_path (str): config file name relative to detectron2's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

    Returns:
        str: a URL to the model
    """
    url = _ModelZooUrls.query(config_path)
    if url is None:
        raise RuntimeError('Pretrained model for {} is not available!'.
            format(config_path))
    return url
