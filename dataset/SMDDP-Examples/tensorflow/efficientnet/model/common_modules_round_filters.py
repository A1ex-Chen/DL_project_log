def round_filters(filters: int, config: dict) ->int:
    """Round number of filters based on width coefficient."""
    width_coefficient = config['width_coefficient']
    min_depth = config['min_depth']
    divisor = config['depth_divisor']
    orig_filters = filters
    if not width_coefficient:
        return filters
    filters *= width_coefficient
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor *
        divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)
