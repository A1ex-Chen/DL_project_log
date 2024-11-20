def load_cache(cache_file):
    with h5py.File(cache_file, 'r') as hf:
        x_train = hf['x_train'][:]
        y_train = hf['y_train'][:]
        x_val = hf['x_val'][:]
        y_val = hf['y_val'][:]
        x_test = hf['x_test'][:]
        y_test = hf['y_test'][:]
        x_labels = [x[0].decode('unicode_escape') for x in hf['x_labels'][:]]
        y_labels = [x[0].decode('unicode_escape') for x in hf['y_labels'][:]]
    return x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels
