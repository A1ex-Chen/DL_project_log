def get_metrics(epss, test_func, dat, **kwargs):
    accs_bar = [test_func(metrics.delta_bar, eps=eps, dat=dat, col_name=
        f'score_dl1_{eps}', **kwargs) for eps in epss]
    accs_l2 = [test_func(metrics.delta_bar, eps=eps, dat=dat, col_name=
        f'score_dl2_{eps}', ord=2, **kwargs) for eps in epss]
    accs_pdy = [test_func(metrics.delta_pdy, dat=dat, col_name='score_de1',
        **kwargs)] * len(epss)
    accs_palldy = [test_func(metrics.delta_palldy, dat=dat, col_name=
        'score_da', **kwargs)] * len(epss)
    accs_pxy = [test_func(metrics.delta_pxd, dat=dat, col_name='score_dx',
        **kwargs)] * len(epss)
    return dat, epss, accs_bar, accs_l2, accs_pdy, accs_palldy, accs_pxy
