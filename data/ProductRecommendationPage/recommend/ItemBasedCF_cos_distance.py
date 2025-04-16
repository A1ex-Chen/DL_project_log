def cos_distance(v1, v2):
    a1 = np.array(v1, dtype=int)
    a2 = np.array(v2, dtype=int)
    a3 = multiply(a1, a2)
    return sum(a3) * 1.0 / (sqrt(sum(a1)) + sqrt(sum(a2)))
