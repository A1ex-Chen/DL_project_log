def _validate_py_syntax(filename):
    with PathManager.open(filename, 'r') as f:
        content = f.read()
    try:
        ast.parse(content)
    except SyntaxError as e:
        raise SyntaxError(f'Config file {filename} has syntax error!') from e
