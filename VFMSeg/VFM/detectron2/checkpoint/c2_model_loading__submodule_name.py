def _submodule_name(key):
    pos = key.rfind('.')
    if pos < 0:
        return None
    prefix = key[:pos + 1]
    return prefix
