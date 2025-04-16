def load_data(params, seed):
    file_train = candle.fetch_file(params['data_url'] + params['train_data'
        ], subdir='Pilot1')
    file_test = candle.fetch_file(params['data_url'] + params['test_data'],
        subdir='Pilot1')
    return candle.load_Xy_data2(file_train, file_test, class_col=[
        'cancer_type'], drop_cols=['case_id', 'cancer_type'], n_cols=params
        ['feature_subsample'], shuffle=params['shuffle'], scaling=params[
        'scaling'], validation_split=params['val_split'], dtype=params[
        'data_type'], seed=seed)
