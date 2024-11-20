@staticmethod
def get_output_shape(oldh: int, oldw: int, short_edge_length: int, max_size:
    int) ->Tuple[int, int]:
    """
        Compute the output size given input size and target short edge length.
        """
    h, w = oldh, oldw
    size = short_edge_length * 1.0
    scale = size / min(h, w)
    if h < w:
        newh, neww = size, scale * w
    else:
        newh, neww = scale * h, size
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return newh, neww
