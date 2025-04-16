def random_colors(N, rgb=False, maximum=255):
    """
    Args:
        N (int): number of unique colors needed
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a list of random_color
    """
    indices = random.sample(range(len(_COLORS)), N)
    ret = [(_COLORS[i] * maximum) for i in indices]
    if not rgb:
        ret = [x[::-1] for x in ret]
    return ret
