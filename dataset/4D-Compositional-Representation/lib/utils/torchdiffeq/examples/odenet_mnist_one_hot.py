def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)
