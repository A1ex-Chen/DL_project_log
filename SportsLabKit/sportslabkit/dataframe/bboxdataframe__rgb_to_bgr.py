def _rgb_to_bgr(color: tuple[int, ...]) ->list[Any]:
    """Convert RGB color to BGR color.

    Args:
        color (tuple): RGB color.

    Returns:
        list: BGR color.
    """
    return list(reversed(color))
