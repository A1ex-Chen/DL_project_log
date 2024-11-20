def _generate_color_palette(n: int) ->np.ndarray:
    """
    Generates random colors for bounding boxes.
    Args:
        n (int): Number of colors to generate.
    Returns:
        np.ndarray: Array of shape (N, 3) containing the colors in RGB format.
    """
    colors = []
    for _ in range(n):
        color = tuple([np.random.randint(0, 256) for _ in range(3)])
        colors.append(color)
    return colors
