def rlt(X, L_0=64, gamma=None, i_max=100):
    """
      This function implements Algorithm 1 for one sample of landmarks.

    Args:
      X: np.array representing the dataset.
      L_0: number of landmarks to sample.
      gamma: float, parameter determining the maximum persistence value.
      i_max: int, upper bound on the value of beta_1 to compute.

    Returns
      An array of size (i_max, ) containing RLT(i, 1, X, L)
      for randomly sampled landmarks.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError('X should be a numpy array')
    if len(X.shape) != 2:
        raise ValueError('X should be 2d array, got shape {}'.format(X.shape))
    N = X.shape[0]
    if gamma is None:
        gamma = 1.0 / 128 * N / 5000
    I_1, alpha_max = witness(X, L_0=L_0, gamma=gamma)
    res = relative(I_1, alpha_max, i_max=i_max)
    return res
