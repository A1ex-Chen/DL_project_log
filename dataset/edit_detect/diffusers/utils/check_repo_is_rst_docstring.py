def is_rst_docstring(docstring):
    """
    Returns `True` if `docstring` is written in rst.
    """
    if _re_rst_special_words.search(docstring) is not None:
        return True
    if _re_double_backquotes.search(docstring) is not None:
        return True
    if _re_rst_example.search(docstring) is not None:
        return True
    return False
