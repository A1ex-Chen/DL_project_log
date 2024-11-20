def load_data(params, seed):
    drop_cols = ['case_id']
    onehot_cols = ['cancer_type']
    y_cols = ['cancer_type']
    if params['use_landmark_genes']:
        lincs_file = 'lincs1000.tsv'
        lincs_path = candle.fetch_file(params['data_url'] + lincs_file,
            'Pilot1')
        df_l1000 = pd.read_csv(lincs_path, sep='\t')
        x_cols = df_l1000['gdc'].tolist()
        drop_cols = None
    else:
        x_cols = None
    train_path = candle.fetch_file(params['data_url'] + params['train_data'
        ], 'Pilot1')
    test_path = candle.fetch_file(params['data_url'] + params['test_data'],
        'Pilot1')
    return candle.load_csv_data(train_path, test_path, x_cols=x_cols,
        y_cols=y_cols, drop_cols=drop_cols, onehot_cols=onehot_cols, n_cols
        =params['feature_subsample'], shuffle=params['shuffle'], scaling=
        params['scaling'], dtype=params['data_type'], validation_split=
        params['val_split'], return_dataframe=False, return_header=True,
        nrows=params['train_samples'] if 'train_samples' in params and 
        params['train_samples'] > 0 else None, seed=seed)
