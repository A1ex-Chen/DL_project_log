def _broadcast_params(params, num_features, name):
    """
    If one size (or aspect ratio) is specified and there are multiple feature
    maps, we "broadcast" anchors of that single size (or aspect ratio)
    over all feature maps.

    If params is list[float], or list[list[float]] with len(params) == 1, repeat
    it num_features time.

    Returns:
        list[list[float]]: param for each feature
    """
    assert isinstance(params, collections.abc.Sequence
        ), f'{name} in anchor generator has to be a list! Got {params}.'
    assert len(params), f'{name} in anchor generator cannot be empty!'
    if not isinstance(params[0], collections.abc.Sequence):
        return [params] * num_features
    if len(params) == 1:
        return list(params) * num_features
    assert len(params
        ) == num_features, f'Got {name} of length {len(params)} in anchor generator, but the number of input features is {num_features}!'
    return params
