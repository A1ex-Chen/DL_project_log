def _group_to_str(group: List[str]) ->str:
    """
    Format a group of parameter name suffixes into a loggable string.
    Args:
        group (list[str]): list of parameter name suffixes.
    Returns:
        str: formated string.
    """
    if len(group) == 0:
        return ''
    if len(group) == 1:
        return '.' + group[0]
    return '.{' + ', '.join(group) + '}'
