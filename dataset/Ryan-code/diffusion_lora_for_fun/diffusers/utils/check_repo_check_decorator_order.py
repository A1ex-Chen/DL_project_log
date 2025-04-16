def check_decorator_order(filename):
    """Check that in the test file `filename` the slow decorator is always last."""
    with open(filename, 'r', encoding='utf-8', newline='\n') as f:
        lines = f.readlines()
    decorator_before = None
    errors = []
    for i, line in enumerate(lines):
        search = _re_decorator.search(line)
        if search is not None:
            decorator_name = search.groups()[0]
            if decorator_before is not None and decorator_name.startswith(
                'parameterized'):
                errors.append(i)
            decorator_before = decorator_name
        elif decorator_before is not None:
            decorator_before = None
    return errors
