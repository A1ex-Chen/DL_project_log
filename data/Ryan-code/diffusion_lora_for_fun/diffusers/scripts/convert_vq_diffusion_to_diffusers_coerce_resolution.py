def coerce_resolution(resolution):
    if isinstance(resolution, int):
        resolution = [resolution, resolution]
    elif isinstance(resolution, (tuple, list)):
        resolution = list(resolution)
    else:
        raise ValueError('Unknown type of resolution:', resolution)
    return resolution
