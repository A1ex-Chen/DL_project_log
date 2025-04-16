def resolve_interpolation_mode(interpolation_type: str):
    """
    Maps a string describing an interpolation function to the corresponding torchvision `InterpolationMode` enum. The
    full list of supported enums is documented at
    https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.functional.InterpolationMode.

    Args:
        interpolation_type (`str`):
            A string describing an interpolation method. Currently, `bilinear`, `bicubic`, `box`, `nearest`,
            `nearest_exact`, `hamming`, and `lanczos` are supported, corresponding to the supported interpolation modes
            in torchvision.

    Returns:
        `torchvision.transforms.InterpolationMode`: an `InterpolationMode` enum used by torchvision's `resize`
        transform.
    """
    if not is_torchvision_available():
        raise ImportError(
            'Please make sure to install `torchvision` to be able to use the `resolve_interpolation_mode()` function.'
            )
    if interpolation_type == 'bilinear':
        interpolation_mode = transforms.InterpolationMode.BILINEAR
    elif interpolation_type == 'bicubic':
        interpolation_mode = transforms.InterpolationMode.BICUBIC
    elif interpolation_type == 'box':
        interpolation_mode = transforms.InterpolationMode.BOX
    elif interpolation_type == 'nearest':
        interpolation_mode = transforms.InterpolationMode.NEAREST
    elif interpolation_type == 'nearest_exact':
        interpolation_mode = transforms.InterpolationMode.NEAREST_EXACT
    elif interpolation_type == 'hamming':
        interpolation_mode = transforms.InterpolationMode.HAMMING
    elif interpolation_type == 'lanczos':
        interpolation_mode = transforms.InterpolationMode.LANCZOS
    else:
        raise ValueError(
            f'The given interpolation mode {interpolation_type} is not supported. Currently supported interpolation modes are `bilinear`, `bicubic`, `box`, `nearest`, `nearest_exact`, `hamming`, and `lanczos`.'
            )
    return interpolation_mode
