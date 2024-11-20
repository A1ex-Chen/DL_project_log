def test_eps_gl(func, dat, eps=0.1, col_name='score', n_smp=None, **kwargs):

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
    dat[col_name] = dat.apply(lambda s: func(*get_probs(s), eps=eps, **
        kwargs), axis=1)
    return np.mean(dat.groupby(['index']).apply(lambda s: s.answer_idx.iloc
        [s[col_name].argmax()]) == 1)
