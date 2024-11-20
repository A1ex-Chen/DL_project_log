def to_json_saveable(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, PosixPath):
        value = str(value)
    return value
