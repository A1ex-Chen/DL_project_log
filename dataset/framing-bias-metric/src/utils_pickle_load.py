def pickle_load(path):
    """pickle.load(path)"""
    with open(path, 'rb') as f:
        return pickle.load(f)
