def save_pickle(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)
