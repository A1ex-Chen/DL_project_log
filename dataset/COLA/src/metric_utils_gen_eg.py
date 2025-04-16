def gen_eg(ds, eps, spacy_model, file_pref=None):
    matched_idx, norm_arr = get_dat_matched_idx(ds, eps=eps).iloc[0]
    cov, premise, interv, outcome, p_dy = ds.iloc[0][['covariates', 'text',
        'interventions', 'outcome', 'p_dy']]

    def tt_ize(s):
        return '\\texttt{' + s + '}' if s is not None else ''

    def small_ize(s):
        return '{\\small ' + s + '}'

    def highlight(s):
        return '\\egtbhlt ' + s
    X = [tt_ize(crop_sent(x.replace('$', '\\$'), spacy_model=spacy_model,
        sent_idx=0, offset=0)) for xidx, x in enumerate(cov)]
    X = [small_ize(f'$\\sfX_{{{xidx + 1}}}$: ' + x) for xidx, x in
        enumerate(X) if len(x) > 0]
    E1 = small_ize(f'$\\sfE_1$: ' + tt_ize(premise))
    E2 = small_ize(f'$\\sfE_2$: ' + tt_ize(outcome))
    D = [small_ize(f'$\\sfA_{{{didx + 1}}}$: ' + tt_ize(d)) for didx, d in
        enumerate(interv)]
    p_dy = [f'${p[0]:.4f}$' for p in p_dy]
    norm_arr = [f'$0$'] + [f'${delta:.4f}$' for delta in norm_arr]
    for mi in matched_idx:
        D[mi] = highlight(D[mi])
    print(f'matched: {len(matched_idx)}, len(X): {len(X)}, len(D): {len(D)}')
    Xstr = '\n'.join([(x + ' & & & \\\\') for x in X])
    Dstr = '\n'.join(['\\, \\\\'] + [(d + ' \\\\') for d in [E1] + D])
    p_dy_str = '\n'.join(['\\, \\\\'] + [(p + ' \\\\') for p in p_dy])
    arr_str = '\n'.join(['\\, \\\\'] + [(delta + ' \\\\') for delta in
        norm_arr])
    if file_pref is not None:
        with open(f'{file_pref}_cov.txt', 'w') as f:
            f.write(Xstr)
        with open(f'{file_pref}_d.txt', 'w') as f:
            f.write(Dstr)
        with open(f'{file_pref}_pdy.txt', 'w') as f:
            f.write(p_dy_str)
        with open(f'{file_pref}_dist.txt', 'w') as f:
            f.write(arr_str)
    return Xstr, Dstr, E2, p_dy_str
