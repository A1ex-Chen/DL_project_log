def relative(I_1, alpha_max, i_max=100):
    """
      For a collection of intervals I_1 this functions computes
      RLT by formulas (2) and (3). This function will be typically called
      on the output of the gudhi persistence_intervals_in_dimension function.

    Args:
      I_1: list of intervals e.g. [[0, 1], [0, 2], [0, np.inf]].
      alpha_max: float, the maximal persistence value
      i_max: int, upper bound on the value of beta_1 to compute.

    Returns
      An array of size (i_max, ) containing desired RLT.
    """
    persistence_intervals = []
    for interval in I_1:
        if not np.isinf(interval[1]):
            persistence_intervals.append(list(interval))
        elif np.isinf(interval[1]):
            persistence_intervals.append([interval[0], alpha_max])
    if len(persistence_intervals) == 0:
        rlt = np.zeros(i_max)
        rlt[0] = 1.0
        return rlt
    persistence_intervals_ext = persistence_intervals + [[0, alpha_max]]
    persistence_intervals_ext = np.array(persistence_intervals_ext)
    persistence_intervals = np.array(persistence_intervals)
    switch_points = np.sort(np.unique(persistence_intervals_ext.flatten()))
    rlt = np.zeros(i_max)
    for i in range(switch_points.shape[0] - 1):
        midpoint = (switch_points[i] + switch_points[i + 1]) / 2
        s = 0
        for interval in persistence_intervals:
            if midpoint >= interval[0] and midpoint < interval[1]:
                s = s + 1
        if s < i_max:
            rlt[s] += switch_points[i + 1] - switch_points[i]
    return rlt / alpha_max
