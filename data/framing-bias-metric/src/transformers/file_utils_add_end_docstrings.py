def add_end_docstrings(*docstr):

    def docstring_decorator(fn):
        fn.__doc__ = fn.__doc__ + ''.join(docstr)
        return fn
    return docstring_decorator
