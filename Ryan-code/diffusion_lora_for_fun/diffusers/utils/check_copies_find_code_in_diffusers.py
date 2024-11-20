def find_code_in_diffusers(object_name):
    """Find and return the code source code of `object_name`."""
    parts = object_name.split('.')
    i = 0
    module = parts[i]
    while i < len(parts) and not os.path.isfile(os.path.join(DIFFUSERS_PATH,
        f'{module}.py')):
        i += 1
        if i < len(parts):
            module = os.path.join(module, parts[i])
    if i >= len(parts):
        raise ValueError(
            f'`object_name` should begin with the name of a module of diffusers but got {object_name}.'
            )
    with open(os.path.join(DIFFUSERS_PATH, f'{module}.py'), 'r', encoding=
        'utf-8', newline='\n') as f:
        lines = f.readlines()
    indent = ''
    line_index = 0
    for name in parts[i + 1:]:
        while line_index < len(lines) and re.search(
            f'^{indent}(class|def)\\s+{name}(\\(|\\:)', lines[line_index]
            ) is None:
            line_index += 1
        indent += '    '
        line_index += 1
    if line_index >= len(lines):
        raise ValueError(
            f' {object_name} does not match any function or class in {module}.'
            )
    start_index = line_index
    while line_index < len(lines) and _should_continue(lines[line_index],
        indent):
        line_index += 1
    while len(lines[line_index - 1]) <= 1:
        line_index -= 1
    code_lines = lines[start_index:line_index]
    return ''.join(code_lines)
