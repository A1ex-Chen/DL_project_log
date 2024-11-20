def stylify(code: str) ->str:
    """
    Applies the ruff part of our `make style` command to some code. This formats the code using `ruff format`.
    As `ruff` does not provide a python api this cannot be done on the fly.

    Args:
        code (`str`): The code to format.

    Returns:
        `str`: The formatted code.
    """
    has_indent = len(get_indent(code)) > 0
    if has_indent:
        code = f'class Bla:\n{code}'
    formatted_code = run_ruff(code)
    return formatted_code[len('class Bla:\n'):
        ] if has_indent else formatted_code
