@staticmethod
def kl_divergence(p, q):
    """
        The Kullback–Leibler divergence.
        Defined only if q != 0 whenever p != 0.
        """
    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(q))
    assert not np.any(np.logical_and(p != 0, q == 0))
    p_pos = p > 0
    return np.sum(p[p_pos] * np.log(p[p_pos] / q[p_pos]))
