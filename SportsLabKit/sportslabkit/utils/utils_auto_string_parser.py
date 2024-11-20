def auto_string_parser(value: str) ->Any:
    """Auxiliary function to parse string values.

    Args:
        value (str): String value to parse.

    Returns:
        value (any): Parsed string value.
    """
    if value.isdigit():
        return int(value)
    if value.replace('.', '', 1).isdigit():
        return float(value)
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    if value.lower() == 'nan':
        return np.nan
    if value.lower() == 'inf':
        return np.inf
    if value.lower() == '-inf':
        return -np.inf
    try:
        return literal_eval(value)
    except (ValueError, SyntaxError):
        pass
    try:
        return dateutil.parser.parse(value)
    except (ValueError, TypeError):
        pass
    return value
