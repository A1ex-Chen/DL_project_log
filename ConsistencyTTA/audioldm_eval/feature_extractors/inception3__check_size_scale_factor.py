def _check_size_scale_factor(dim):
    if size is None and scale_factor is None:
        raise ValueError('either size or scale_factor should be defined')
    if size is not None and scale_factor is not None:
        raise ValueError('only one of size or scale_factor should be defined')
    if scale_factor is not None and isinstance(scale_factor, tuple) and len(
        scale_factor) != dim:
        raise ValueError(
            'scale_factor shape must match input shape. Input is {}D, scale_factor size is {}'
            .format(dim, len(scale_factor)))
