def calc_se(p1, n1, p2, n2):
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    return np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
