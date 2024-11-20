def load_by_drug_data(drug='1', cell_features=['expression'], shuffle=True,
    use_gi50=False, logconc=-4.0, subsample='naive_balancing',
    feature_subsample=None, scaling='std', scramble=False, verbose=True):
    """Load dataframe for by drug models

    Parameters
    ----------
    drug: drug NSC ID
    cell_features: list of strings from 'expression', 'expression_5platform', 'mirna', 'proteome', 'all' (default ['expression'])
        use one or more cell line feature sets: gene expression, microRNA, proteome
        use 'all' for ['expression', 'mirna', 'proteome']
    shuffle : True or False, optional (default True)
        if True shuffles the merged data before splitting training and validation sets
    scramble: True or False, optional (default False)
        if True randomly shuffle dose response data as a control
    use_gi50: True of False, optional (default False)
        use NCI GI50 value instead of percent growth at log concentration levels
    logconc: float value between -3 and -7, optional (default -4.)
        log concentration of drug to return cell line growth
    feature_subsample: None or integer (default None)
        number of feature columns to use from cellline expressions and drug descriptors
    scaling: None, 'std', 'minmax' or 'maxabs' (default 'std')
        type of feature scaling: 'maxabs' to [-1,1], 'maxabs' to [-1, 1], 'std' for standard normalization
    subsample: 'naive_balancing' or None
        if True balance dose response data with crude subsampling
    scramble: True or False, optional (default False)
        if True randomly shuffle dose response data as a control
    """
    if 'all' in cell_features:
        cell_features = ['expression', 'mirna', 'proteome']
    df_resp = load_dose_response(subsample=subsample, min_logconc=logconc,
        max_logconc=logconc, fraction=True)
    df_resp = df_resp.reset_index()
    df = df_resp[df_resp['NSC'] == drug]
    df = df[['CELLNAME', 'GROWTH']]
    input_dims = collections.OrderedDict()
    for fea in cell_features:
        if fea == 'expression' or fea == 'expression_u133p2':
            df_expr_u133p2 = load_cell_expression_u133p2(ncols=
                feature_subsample, scaling=scaling)
            df = df.merge(df_expr_u133p2, on='CELLNAME')
            input_dims['expression_u133p2'] = df_expr_u133p2.shape[1] - 1
        elif fea == 'expression_5platform':
            df_expr_5p = load_cell_expression_5platform(ncols=
                feature_subsample, scaling=scaling)
            df = df.merge(df_expr_5p, on='CELLNAME')
            input_dims['expression_5platform'] = df_expr_5p.shape[1] - 1
        elif fea == 'mirna':
            df_mirna = load_cell_mirna(ncols=feature_subsample, scaling=scaling
                )
            df = df.merge(df_mirna, on='CELLNAME')
            input_dims['microRNA'] = df_mirna.shape[1] - 1
        elif fea == 'proteome':
            df_prot = load_cell_proteome(ncols=feature_subsample, scaling=
                scaling)
            df = df.merge(df_prot, on='CELLNAME')
            input_dims['proteome'] = df_prot.shape[1] - 1
    df = df.set_index('CELLNAME')
    if df.shape[0] and verbose:
        print('Loaded {} rows and {} columns'.format(df.shape[0], df.shape[1]))
        print('Input features:', ', '.join(['{}: {}'.format(k, v) for k, v in
            input_dims.items()]))
    return df
