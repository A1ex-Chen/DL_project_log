def get_image_size_from_model(arch):
    """If the given model has a preferred image size, return it."""
    if 'efficientnet' in arch:
        efficientnet_name = arch
        if efficientnet_name in efficientnet_model.MODEL_CONFIGS:
            return efficientnet_model.MODEL_CONFIGS[efficientnet_name][
                'resolution']
    return None
