def split_string(string, separators):
    """
    Function to split a given string based on a list of separators.

    Args:
    string (str): The input string to be split.
    separators (list): A list of separators to be used for splitting the string.

    Returns:
    A list containing the split string with separators included.
    """
    pattern = '|'.join(re.escape(separator) for separator in separators)
    result = re.split(f'({pattern})', string)
    return [elem for elem in result if elem]
