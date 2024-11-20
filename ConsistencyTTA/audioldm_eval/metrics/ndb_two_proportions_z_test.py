@staticmethod
def two_proportions_z_test(p1, n1, p2, n2, significance_level, z_threshold=None
    ):
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    if z_threshold is not None:
        return abs(z) > z_threshold
    p_values = 2.0 * norm.cdf(-1.0 * np.abs(z))
    return p_values < significance_level
