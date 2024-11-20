def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, 'wb') as f:
        return pickle.dump(obj, f)
