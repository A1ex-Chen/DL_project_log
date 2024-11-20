def get_code(mod):
    try:
        return _get_script_mod(mod)._c.code
    except AttributeError:
        pass
    try:
        return mod.code
    except AttributeError:
        return None
