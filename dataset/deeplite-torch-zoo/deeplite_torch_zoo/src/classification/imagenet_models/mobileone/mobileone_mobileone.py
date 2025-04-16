def mobileone(num_classes: int=1000, inference_mode: bool=False, variant:
    str='s0') ->nn.Module:
    """Get MobileOne model.

    :param num_classes: Number of classes in the dataset.
    :param inference_mode: If True, instantiates model in inference mode.
    :param variant: Which type of model to generate.
    :return: MobileOne model."""
    variant_params = PARAMS[variant]
    return MobileOne(num_classes=num_classes, inference_mode=inference_mode,
        **variant_params)
