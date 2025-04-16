def get_probs(s):
    p_xd, p_dy, p_xy = s[['p_xd', 'p_dy', 'p_xy']]
    p_dy = np.array(p_dy)
    p_xd = np.array(p_xd)
    p_xy = np.array(p_xy)
    if n_smp is None:
        return p_xd, p_dy, p_xy
    nd = p_xy.shape[0]
    smp_indices = np.random.choice(np.arange(nd), np.minimum(nd, n_smp),
        replace=False)
    return p_xd[smp_indices], p_dy, p_xy[smp_indices]
