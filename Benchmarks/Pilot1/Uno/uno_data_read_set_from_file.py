def read_set_from_file(path):
    if path:
        with open(path, 'r') as f:
            text = f.read().strip()
            subset = text.split()
    else:
        subset = None
    return subset
