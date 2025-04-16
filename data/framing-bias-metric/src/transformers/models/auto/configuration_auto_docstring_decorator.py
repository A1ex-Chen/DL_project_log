def docstring_decorator(fn):
    docstrings = fn.__doc__
    lines = docstrings.split('\n')
    i = 0
    while i < len(lines) and re.search('^(\\s*)List options\\s*$', lines[i]
        ) is None:
        i += 1
    if i < len(lines):
        indent = re.search('^(\\s*)List options\\s*$', lines[i]).groups()[0]
        if use_model_types:
            indent = f'{indent}    '
        lines[i] = _list_model_options(indent, config_to_class=
            config_to_class, use_model_types=use_model_types)
        docstrings = '\n'.join(lines)
    else:
        raise ValueError(
            f"""The function {fn} should have an empty 'List options' in its docstring as placeholder, current docstring is:
{docstrings}"""
            )
    fn.__doc__ = docstrings
    return fn
