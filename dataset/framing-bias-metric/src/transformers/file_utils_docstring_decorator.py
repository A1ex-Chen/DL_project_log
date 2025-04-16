def docstring_decorator(fn):
    docstrings = fn.__doc__
    lines = docstrings.split('\n')
    i = 0
    while i < len(lines) and re.search('^\\s*Returns?:\\s*$', lines[i]
        ) is None:
        i += 1
    if i < len(lines):
        lines[i] = _prepare_output_docstrings(output_type, config_class)
        docstrings = '\n'.join(lines)
    else:
        raise ValueError(
            f"""The function {fn} should have an empty 'Return:' or 'Returns:' in its docstring as placeholder, current docstring is:
{docstrings}"""
            )
    fn.__doc__ = docstrings
    return fn
