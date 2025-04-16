def docstring_decorator(fn):
    func_doc = fn.__doc__
    lines = func_doc.split('\n')
    i = 0
    while i < len(lines) and re.search('^\\s*Examples?:\\s*$', lines[i]
        ) is None:
        i += 1
    if i < len(lines):
        lines[i] = example_docstring
        func_doc = '\n'.join(lines)
    else:
        raise ValueError(
            f"""The function {fn} should have an empty 'Examples:' in its docstring as placeholder, current docstring is:
{func_doc}"""
            )
    fn.__doc__ = func_doc
    return fn
