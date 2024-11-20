def fetch_data(params):
    data_url = params['data_url']
    if params['data_dir'] is None:
        params['data_dir'] = candle.fetch_file(data_url + params[
            'train_data'], subdir='Examples/histogen')
    else:
        tempfile = candle.fetch_file(data_url + params['train_data'],
            subdir='Examples/histogen')
        params['data_dir'] = os.path.join(os.path.dirname(tempfile), params
            ['data_dir'])
