@staticmethod
def deserialize(path_to_pickle_file: str) ->'Config':
    with open(path_to_pickle_file, 'rb') as f:
        config = pickle.load(f)
    return config
