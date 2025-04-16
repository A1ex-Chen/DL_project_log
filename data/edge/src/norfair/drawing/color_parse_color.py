def parse_color(color_like: ColorLike) ->ColorType:
    """Makes best effort to parse the given value to a Color

    Parameters
    ----------
    color_like : ColorLike
        Can be one of:

        1. a string with the 6 digits hex value (`"#ff0000"`)
        2. a string with one of the names defined in Colors (`"red"`)
        3. a BGR tuple (`(0, 0, 255)`)

    Returns
    -------
    Color
        The BGR tuple.
    """
    if isinstance(color_like, str):
        if color_like.startswith('#'):
            return hex_to_bgr(color_like)
        else:
            return getattr(Color, color_like)
    return tuple([int(v) for v in color_like])
