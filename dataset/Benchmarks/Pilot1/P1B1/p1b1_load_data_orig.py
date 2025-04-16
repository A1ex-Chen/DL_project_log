def load_data_orig(params, seed):
    if params['with_type']:
        drop_cols = ['case_id']
        onehot_cols = ['cancer_type']
    else:
        drop_cols = ['case_id', 'cancer_type']
        onehot_cols = None
    if params['use_landmark_genes']:
        lincs_file = 'lincs1000.tsv'
        lincs_path = candle.fetch_file(params['data_url'] + lincs_file)
        df_l1000 = pd.read_csv(lincs_path, sep='\t')
        usecols = df_l1000['gdc']
        drop_cols = None
    else:
        usecols = None
    return candle.load_X_data(params['data_url'], params['train_data'],
        params['test_data'], drop_cols=drop_cols, onehot_cols=onehot_cols,
        usecols=usecols, n_cols=params['feature_subsample'], shuffle=params
        ['shuffle'], scaling=params['scaling'], validation_split=params[
        'val_split'], dtype=params['data_type'], seed=seed)
