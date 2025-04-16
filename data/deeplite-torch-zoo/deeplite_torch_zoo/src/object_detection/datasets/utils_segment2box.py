def segment2box(segment, width=640, height=640):
    """
    Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)

    Args:
      segment (torch.Tensor): the segment label
      width (int): the width of the image. Defaults to 640
      height (int): The height of the image. Defaults to 640

    Returns:
      (np.ndarray): the minimum and maximum x and y values of the segment.
    """
    x, y = segment.T
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype
        ) if any(x) else np.zeros(4, dtype=segment.dtype)
