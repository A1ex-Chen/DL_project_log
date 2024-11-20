def _get_indent(t):
    """Returns the indentation in the first line of t"""
    search = re.search('^(\\s*)\\S', t)
    return '' if search is None else search.groups()[0]
