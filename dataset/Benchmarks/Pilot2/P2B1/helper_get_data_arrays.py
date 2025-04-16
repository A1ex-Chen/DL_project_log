def get_data_arrays(f):
    data = np.load(f)
    X = data['features']
    nbrs = data['neighbors']
    resnums = data['resnums']
    return X, nbrs, resnums
