def _import_module(module_name, warn=True, prefix='_py_', ignore='_'):
    """Try import all public attributes from module into global namespace.

    Existing attributes with name clashes are renamed with prefix.
    Attributes starting with underscore are ignored by default.

    Return True on successful import.

    """
    try:
        module = __import__(module_name)
    except ImportError:
        if warn:
            warnings.warn('Failed to import module ' + module_name)
    else:
        for attr in dir(module):
            if ignore and attr.startswith(ignore):
                continue
            if prefix:
                if attr in globals():
                    globals()[prefix + attr] = globals()[attr]
                elif warn:
                    warnings.warn('No Python implementation of ' + attr)
            globals()[attr] = getattr(module, attr)
        return True
