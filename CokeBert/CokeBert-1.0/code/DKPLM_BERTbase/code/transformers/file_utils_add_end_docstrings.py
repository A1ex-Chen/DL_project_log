def add_end_docstrings(*docstr):

    def docstring_decorator(fn):
        return fn
    return docstring_decorator
