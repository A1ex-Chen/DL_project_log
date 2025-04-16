def get_metrics_todf(epss, test_func, dat, **kwargs):
    accs_bar = [test_func(metrics.delta_bar, eps=eps, dat=dat, col_name=
        'tmp_score', **kwargs) for eps in epss]
    accs_l2 = [test_func(metrics.delta_bar, eps=eps, dat=dat, col_name=
        'tmp_score', ord=2, **kwargs) for eps in epss]
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
