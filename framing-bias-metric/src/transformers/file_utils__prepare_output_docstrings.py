def _prepare_output_docstrings(output_type, config_class):
    """
    Prepares the return part of the docstring using `output_type`.
    """
    docstrings = output_type.__doc__
    lines = docstrings.split('\n')
    i = 0
    while i < len(lines) and re.search('^\\s*(Args|Parameters):\\s*$', lines[i]
        ) is None:
        i += 1
    if i < len(lines):
        docstrings = '\n'.join(lines[i + 1:])
        docstrings = _convert_output_args_doc(docstrings)
    full_output_type = f'{output_type.__module__}.{output_type.__name__}'
    intro = TF_RETURN_INTRODUCTION if output_type.__name__.startswith('TF'
        ) else PT_RETURN_INTRODUCTION
    intro = intro.format(full_output_type=full_output_type, config_class=
        config_class)
    return intro + docstrings
