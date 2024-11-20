def corners_2d(dims, origin=0.5):
    """generate relative 2d box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, 2]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
        dtype (output dtype, optional): Defaults to np.float32

    Returns:
        float array, shape=[N, 4, 2]: returned corners.
        point layout: x0y0, x0y1, x1y1, x1y0
    """
    return corners_nd(dims, origin)
