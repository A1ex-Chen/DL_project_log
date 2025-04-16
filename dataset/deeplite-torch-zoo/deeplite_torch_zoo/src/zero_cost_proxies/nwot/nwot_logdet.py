def logdet(K):
    s, ld = np.linalg.slogdet(K)
    return 0 if np.isneginf(ld) and s == 0 else ld
