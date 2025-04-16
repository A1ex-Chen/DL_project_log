def check_all_decorator_order():
    """Check that in all test files, the slow decorator is always last."""
    errors = []
    for fname in os.listdir(PATH_TO_TESTS):
        if fname.endswith('.py'):
            filename = os.path.join(PATH_TO_TESTS, fname)
            new_errors = check_decorator_order(filename)
            errors += [f'- {filename}, line {i}' for i in new_errors]
    if len(errors) > 0:
        msg = '\n'.join(errors)
        raise ValueError(
            f"""The parameterized decorator (and its variants) should always be first, but this is not the case in the following files:
{msg}"""
            )
