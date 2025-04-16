def load_data(params):
    train_path = candle.fetch_file(params['data_url'] + params['train_data'
        ], 'Pilot1')
    test_path = candle.fetch_file(params['data_url'] + params['test_data'],
        'Pilot1')
    if params['feature_subsample'] > 0:
        usecols = list(range(params['feature_subsample']))
    else:
        usecols = None
    return candle.load_Xy_data_noheader(train_path, test_path, params[
        'classes'], usecols, scaling='maxabs', dtype=params['data_type'])
