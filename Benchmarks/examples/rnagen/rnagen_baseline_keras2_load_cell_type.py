def load_cell_type(gParams):
    link = gParams['data_url'] + gParams['train_data']
    path = candle.fetch_file(link, subdir='Examples')
    df = pd.read_csv(path, engine='c', sep='\t', header=None)
    df.columns = ['Sample', 'type']
    return df
