def random_landmarks(X, L_0=32):
    """
    Randomly sample L_0 points from X.
    """
    sz = X.shape[0]
    idx = np.random.choice(sz, L_0)
    L = X[idx]
    return L
