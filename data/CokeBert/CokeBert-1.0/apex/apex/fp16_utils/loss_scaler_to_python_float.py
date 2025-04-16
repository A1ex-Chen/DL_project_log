def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]
