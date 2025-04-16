def get_distance_by_name(name: str) ->Distance:
    """
    Select a distance by name.

    Parameters
    ----------
    name : str
        A string defining the metric to get.

    Returns
    -------
    Distance
        The distance object.
    """
    if name in _SCALAR_DISTANCE_FUNCTIONS:
        warning(
            f'You are using a scalar distance function. If you want to speed up the tracking process please consider using a vectorized distance function such as {AVAILABLE_VECTORIZED_DISTANCES}.'
            )
        distance = _SCALAR_DISTANCE_FUNCTIONS[name]
        distance_function = ScalarDistance(distance)
    elif name in _SCIPY_DISTANCE_FUNCTIONS:
        distance_function = ScipyDistance(name)
    elif name in _VECTORIZED_DISTANCE_FUNCTIONS:
        if name == 'iou_opt':
            warning('iou_opt is deprecated, use iou instead')
        distance = _VECTORIZED_DISTANCE_FUNCTIONS[name]
        distance_function = VectorizedDistance(distance)
    else:
        raise ValueError(
            f"Invalid distance '{name}', expecting one of {list(_SCALAR_DISTANCE_FUNCTIONS.keys()) + AVAILABLE_VECTORIZED_DISTANCES}"
            )
    return distance_function
