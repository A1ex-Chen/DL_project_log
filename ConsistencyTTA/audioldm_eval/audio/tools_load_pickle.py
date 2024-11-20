def load_pickle(fname):
    with open(fname, 'rb') as f:
        res = pickle.load(f)
    return res
