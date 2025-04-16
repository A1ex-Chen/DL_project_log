def func_load(code, defaults=None, closure=None, globs=None):
    """Deserialize user defined function."""
    if isinstance(code, (tuple, list)):
        code, defaults, closure = code
    code = marshal.loads(code.encode('raw_unicode_escape'))
    if closure is not None:
        closure = func_reconstruct_closure(closure)
    if globs is None:
        globs = globals()
    return python_types.FunctionType(code, globs, name=code.co_name,
        argdefs=defaults, closure=closure)
