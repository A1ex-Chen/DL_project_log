def ind_fn(t, b, c, d):
    numerator = t * (np.log(c) * (np.log(d) - np.log(t) + 1) - np.log(d) *
        np.log(t) + np.log(d) + np.log(t) ** 2 - 2 * np.log(t) + 2)
    denominator = (np.log(b) - np.log(c)) * (np.log(b) - np.log(d))
    return numerator / denominator
