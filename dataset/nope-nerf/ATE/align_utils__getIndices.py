def _getIndices(n_aligned, total_n):
    if n_aligned == -1:
        idxs = np.arange(0, total_n)
    else:
        assert n_aligned <= total_n and n_aligned >= 1
        idxs = np.arange(0, n_aligned)
    return idxs
