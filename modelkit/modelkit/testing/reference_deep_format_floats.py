def deep_format_floats(obj, depth=5):
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, dict):
        return {k: deep_format_floats(v, depth) for k, v in obj.items()}
    elif isinstance(obj, Iterable):
        return type(obj)(deep_format_floats(v, depth) for v in obj)
    elif isinstance(obj, float):
        return ('{:.' + str(depth) + 'f}').format(obj)
    else:
        return obj
