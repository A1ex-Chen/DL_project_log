def load_p(filename):
    import pickle
    with open(filename, 'rb') as file:
        z = pickle.load(file)
    return z
