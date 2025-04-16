def get_metrics_todf_mult(epss, test_func, dat, n_workers=30, **kwargs):

    def dl1_func(eps):
        return test_func(metrics.delta_bar, eps=eps, dat=dat, col_name=
            'tmp_score', **kwargs)

    def dl2_func(eps):
        return test_func(metrics.delta_bar, eps=eps, dat=dat, col_name=
            'tmp_score', ord=None, **kwargs)
    with pathos.multiprocessing.ProcessingPool(n_workers) as pool:
        accs_bar = pool.map(dl1_func, epss)
        accs_l2 = pool.map(dl2_func, epss)
    df1 = pd.DataFrame(np.array([epss, accs_bar]).T, columns=['eps', 'acc'])
    df2 = pd.DataFrame(np.array([epss, accs_l2]).T, columns=['eps', 'acc'])
    df1['score'] = 'score_dl1'
    df2['score'] = 'score_dl2'
    df = pd.concat([df1, df2])
    accs_pdy = test_func(metrics.delta_pdy, dat=dat, col_name='tmp_score',
        **kwargs)
    accs_palldy = test_func(metrics.delta_palldy, dat=dat, col_name=
        'tmp_score', **kwargs)
    accs_pxy = test_func(metrics.delta_pxd, dat=dat, col_name='tmp_score',
        **kwargs)
    df = pd.concat([df, pd.DataFrame([[-1, accs_pdy, 'score_de1'], [-1,
        accs_palldy, 'score_da'], [-1, accs_pxy, 'score_dx']], columns=[
        'eps', 'acc', 'score'])])
    return df
