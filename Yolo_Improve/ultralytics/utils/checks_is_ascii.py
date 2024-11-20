def is_ascii(s) ->bool:
    """
    Check if a string is composed of only ASCII characters.

    Args:
        s (str): String to be checked.

    Returns:
        (bool): True if the string is composed only of ASCII characters, False otherwise.
    """
    s = str(s)
    return all(ord(c) < 128 for c in s)
