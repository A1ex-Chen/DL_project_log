def lmrk_table(W, L):
    """
      Helper function to construct an input for the gudhi.WitnessComplex
      function.

    Args:
      W: 2d array of size w x d, containing witnesses
      L: 2d array of size l x d containing landmarks

    Returns
      Return a 3d array D of size w x l x 2 and the maximal distance
      between W and L.

      D satisfies the property that D[i, :, :] is [idx_i, dists_i],
      where dists_i are the sorted distances from the i-th witness to each
      point in L and idx_i are the indices of the corresponding points
      in L, e.g.,
      D[i, :, :] = [[0, 0.1], [1, 0.2], [3, 0.3], [2, 0.4]]
    """
    a = cdist(W, L)
    max_val = np.max(a)
    idx = np.argsort(a)
    b = a[np.arange(np.shape(a)[0])[:, np.newaxis], idx]
    return np.dstack([idx, b]), max_val
