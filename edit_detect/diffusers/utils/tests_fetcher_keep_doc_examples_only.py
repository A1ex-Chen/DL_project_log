def keep_doc_examples_only(content: str) ->str:
    """
    Remove everything from the code content except the doc examples (used to determined if a diff should trigger doc
    tests or not).

    Args:
        content (`str`): The code to clean

    Returns:
        `str`: The cleaned code.
    """
    splits = content.split('```')
    content = '```' + '```'.join(splits[1::2]) + '```'
    lines_to_keep = []
    for line in content.split('\n'):
        line = re.sub('#.*$', '', line)
        if len(line) != 0 and not line.isspace():
            lines_to_keep.append(line)
    return '\n'.join(lines_to_keep)
