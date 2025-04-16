def load_by_cell_data(cell='BR:MCF7', drug_features=['descriptors'],
    shuffle=True, min_logconc=-5.0, max_logconc=-4.0, subsample=
    'naive_balancing', feature_subsample=None, scaling='std', scramble=
    False, verbose=True):
    """Load dataframe for by cellline models

    Parameters
    ----------
    cell: cellline ID
    drug_features: list of strings from 'descriptors', 'latent', 'all', 'noise' (default ['descriptors'])
        use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder
        trained on NSC drugs, or both; use random features if set to noise
    shuffle : True or False, optional (default True)
        if True shuffles the merged data before splitting training and validation sets
    scramble: True or False, optional (default False)
        if True randomly shuffle dose response data as a control
    min_logconc: float value between -3 and -7, optional (default -5.)
        min log concentration of drug to return cell line growth
    max_logconc: float value between -3 and -7, optional (default -4.)
        max log concentration of drug to return cell line growth
    feature_subsample: None or integer (default None)
        number of feature columns to use from cellline expressions and drug descriptors
    scaling: None, 'std', 'minmax' or 'maxabs' (default 'std')
        type of feature scaling: 'maxabs' to [-1,1], 'maxabs' to [-1, 1], 'std' for standard normalization
    subsample: 'naive_balancing' or None
        if True balance dose response data with crude subsampling
    """
    if 'all' in drug_features:
        drug_features = ['descriptors', 'latent']
    df_resp = load_dose_response(subsample=subsample, min_logconc=
        min_logconc, max_logconc=max_logconc, fraction=True)
    df = df_resp[df_resp['CELLNAME'] == cell].reset_index()
    df = df[['NSC', 'GROWTH', 'LOG_CONCENTRATION']]
    df = df.rename(columns={'LOG_CONCENTRATION': 'LCONC'})
    input_dims = collections.OrderedDict()
    input_dims['log_conc'] = 1
    for fea in drug_features:
        if fea == 'descriptors':
            df_desc = load_drug_descriptors(ncols=feature_subsample,
                scaling=scaling)
            df = df.merge(df_desc, on='NSC')
            input_dims['drug_descriptors'] = df_desc.shape[1] - 1
        elif fea == 'latent':
            df_ag = load_drug_autoencoded_AG(ncols=feature_subsample,
                scaling=scaling)
            df = df.merge(df_ag, on='NSC')
            input_dims['smiles_latent_AG'] = df_ag.shape[1] - 1
        elif fea == 'noise':
            df_drug_ids = df[['NSC']].drop_duplicates()
            noise = np.random.normal(size=(df_drug_ids.shape[0], 500))
            df_rand = pd.DataFrame(noise, index=df_drug_ids['NSC'], columns
                =['RAND-{:03d}'.format(x) for x in range(500)])
            df = df.merge(df_rand, on='NSC')
            input_dims['drug_noise'] = df_rand.shape[1] - 1
    df = df.set_index('NSC')
    if df.shape[0] and verbose:
        print('Loaded {} rows and {} columns'.format(df.shape[0], df.shape[1]))
        print('Input features:', ', '.join(['{}: {}'.format(k, v) for k, v in
            input_dims.items()]))
    return df
