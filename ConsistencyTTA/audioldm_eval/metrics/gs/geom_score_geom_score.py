def geom_score(rlts1, rlts2):
    """
      This function implements Algorithm 2.

    Args:
       rlts1 and rlts2: arrays as returned by the function "rlts".
    Returns
       Float, a number representing topological similarity of two datasets.

    """
    mrlt1 = np.mean(rlts1, axis=0)
    mrlt2 = np.mean(rlts2, axis=0)
    return np.sum((mrlt1 - mrlt2) ** 2)
