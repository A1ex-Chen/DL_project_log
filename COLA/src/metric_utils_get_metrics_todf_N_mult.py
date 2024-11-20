def get_metrics_todf_N_mult(epss_input, test_func, dat, n_smp, n_rep=50,
    n_workers=30, **kwargs):

    def dl1_func(eps):
        return test_func(metrics.delta_bar, eps=eps, dat=dat, col_name=
            'tmp_score', n_smp=n_smp, **kwargs)

    def dl2_func(eps):
        return test_func(metrics.delta_bar, eps=eps, dat=dat, col_name=
            'tmp_score', ord=None, n_smp=n_smp, **kwargs)

    def pdy_func(dummy):
        return test_func(metrics.delta_pdy, dat=dat, col_name='tmp_score',
            n_smp=n_smp, **kwargs)

    def palldy_func(dummy):
        return test_func(metrics.delta_palldy, dat=dat, col_name=
            'tmp_score', n_smp=n_smp, **kwargs)

    def pxy_func(dummy):
        return test_func(metrics.delta_pxd, dat=dat, col_name='tmp_score',
            n_smp=n_smp, **kwargs)

    def get_df(accs, eps, score_lb):
        df = pd.DataFrame(accs, columns=['acc'])
        df['eps'] = eps
        df['score'] = score_lb
        return df
    epss = np.repeat(epss_input, n_rep)
    dummies = np.arange(n_smp)
    with pathos.multiprocessing.ProcessingPool(n_workers) as pool:
        accs_bar = pool.map(dl1_func, epss)
        accs_l2 = pool.map(dl2_func, epss)
        accs_pdy = pool.map(pdy_func, dummies)
        accs_palldy = pool.map(palldy_func, dummies)
        accs_pxy = pool.map(pxy_func, dummies)
    df1 = pd.DataFrame(np.array([epss, accs_bar]).T, columns=['eps', 'acc'])
    df2 = pd.DataFrame(np.array([epss, accs_l2]).T, columns=['eps', 'acc'])
    df1['score'] = 'score_dl1'
    df2['score'] = 'score_dl2'
    df = pd.concat([df1, df2])
    df = pd.concat([df, get_df(accs_pdy, -1, 'score_de1'), get_df(
        accs_palldy, -1, 'score_da'), get_df(accs_pxy, -1, 'score_dx')])
    return df
