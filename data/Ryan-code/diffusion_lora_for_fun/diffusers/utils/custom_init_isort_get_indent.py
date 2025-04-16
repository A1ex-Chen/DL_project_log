def get_indent(line: str) ->str:
    """Returns the indent in  given line (as string)."""
    search = _re_indent.search(line)
    return '' if search is None else search.groups()[0]
