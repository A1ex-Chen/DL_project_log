def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return '.'.join(path.split('.')[n_shave_prefix_segments:])
    else:
        return '.'.join(path.split('.')[:n_shave_prefix_segments])
