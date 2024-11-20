def tokenize_numbers(text_array: List[str]) ->List[str]:
    """
    Splits large comma-separated numbers and floating point values. This is done by replacing commas with ' @,@ ' and
    dots with ' @.@ '.

    Args:
        text_array: An already tokenized text as list.

    Returns:
        A list of strings with tokenized numbers.

    Example::
        >>> tokenize_numbers(["$", "5,000", "1.73", "m"])
        ["$", "5", "@,@", "000", "1", "@.@", "73", "m"]
    """
    tokenized = []
    for i in range(len(text_array)):
        reg, sub = MATCH_NUMBERS
        replaced = re.sub(reg, sub, text_array[i]).split()
        tokenized.extend(replaced)
    return tokenized
