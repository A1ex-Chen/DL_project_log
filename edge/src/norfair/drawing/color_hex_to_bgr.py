def hex_to_bgr(hex_value: str) ->ColorType:
    """Converts conventional 6 digits hex colors to BGR tuples

    Parameters
    ----------
    hex_value : str
        hex value with leading `#` for instance `"#ff0000"`

    Returns
    -------
    Tuple[int, int, int]
        BGR values

    Raises
    ------
    ValueError
        if the string is invalid
    """
    if re.match('#[a-f0-9]{6}$', hex_value):
        return int(hex_value[5:7], 16), int(hex_value[3:5], 16), int(hex_value
            [1:3], 16)
    if re.match('#[a-f0-9]{3}$', hex_value):
        return int(hex_value[3] * 2, 16), int(hex_value[2] * 2, 16), int(
            hex_value[1] * 2, 16)
    raise ValueError(f"'{hex_value}' is not a valid color")
