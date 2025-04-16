def detokenize_numbers(text: str) ->str:
    """
    Inverts the operation of `tokenize_numbers`. This is replacing ' @,@ ' and ' @.@' by ',' and '.'.

    Args:
        text: A string where the number should be detokenized.

    Returns:
        A detokenized string.

    Example::
        >>> detokenize_numbers("$ 5 @,@ 000 1 @.@ 73 m")
        "$ 5,000 1.73 m"
    """
    for reg, sub in DETOKENIZE_NUMBERS:
        text = re.sub(reg, sub, text)
    return text
