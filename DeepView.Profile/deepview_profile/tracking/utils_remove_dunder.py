def remove_dunder(fn_name):
    match = DUNDER_REGEX.match(fn_name)
    if match is None:
        return fn_name
    return match.group('name')
