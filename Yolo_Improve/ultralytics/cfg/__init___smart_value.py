def smart_value(v):
    """Convert a string to its appropriate type (int, float, bool, None, etc.)."""
    v_lower = v.lower()
    if v_lower == 'none':
        return None
    elif v_lower == 'true':
        return True
    elif v_lower == 'false':
        return False
    else:
        with contextlib.suppress(Exception):
            return eval(v)
        return v
