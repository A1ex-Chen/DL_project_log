def remove_colorstr(input_string):
    """
    Removes ANSI escape codes from a string, effectively un-coloring it.

    Args:
        input_string (str): The string to remove color and style from.

    Returns:
        (str): A new string with all ANSI escape codes removed.

    Examples:
        >>> remove_colorstr(colorstr('blue', 'bold', 'hello world'))
        >>> 'hello world'
    """
    ansi_escape = re.compile('\\x1B\\[[0-9;]*[A-Za-z]')
    return ansi_escape.sub('', input_string)
