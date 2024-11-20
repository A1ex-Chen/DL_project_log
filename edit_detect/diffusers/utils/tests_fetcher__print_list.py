def _print_list(l) ->str:
    """
    Pretty print a list of elements with one line per element and a - starting each line.
    """
    return '\n'.join([f'- {f}' for f in l])
