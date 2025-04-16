def clean_code(content: str) ->str:
    """
    Remove docstrings, empty line or comments from some code (used to detect if a diff is real or only concern
    comments or docstings).

    Args:
        content (`str`): The code to clean

    Returns:
        `str`: The cleaned code.
    """
    splits = content.split('"""')
    content = ''.join(splits[::2])
    splits = content.split("'''")
    content = ''.join(splits[::2])
    lines_to_keep = []
    for line in content.split('\n'):
        line = re.sub('#.*$', '', line)
        if len(line) != 0 and not line.isspace():
            lines_to_keep.append(line)
    return '\n'.join(lines_to_keep)
