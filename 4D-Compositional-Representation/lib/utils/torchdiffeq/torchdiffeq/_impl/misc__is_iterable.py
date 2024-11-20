def _is_iterable(inputs):
    try:
        iter(inputs)
        return True
    except TypeError:
        return False
