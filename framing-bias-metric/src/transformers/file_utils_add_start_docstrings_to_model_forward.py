def add_start_docstrings_to_model_forward(*docstr):

    def docstring_decorator(fn):
        class_name = ':class:`~transformers.{}`'.format(fn.__qualname__.
            split('.')[0])
        intro = (
            '   The {} forward method, overrides the :func:`__call__` special method.'
            .format(class_name))
        note = """

    .. note::
        Although the recipe for forward pass needs to be defined within this function, one should call the
        :class:`Module` instance afterwards instead of this since the former takes care of running the pre and post
        processing steps while the latter silently ignores them.
        """
        fn.__doc__ = intro + note + ''.join(docstr) + (fn.__doc__ if fn.
            __doc__ is not None else '')
        return fn
    return docstring_decorator
