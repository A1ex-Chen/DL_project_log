def load_data(params):
    data_path_train = candle.fetch_file(params['data_url'] + params[
        'train_data'], 'Pilot1')
    data_path_val = candle.fetch_file(params['data_url'] + params[
        'val_data'], 'Pilot1')
    vocab_size = params['vocab_size']
    maxlen = params['maxlen']
    data_train = pd.read_csv(data_path_train)
    data_vali = pd.read_csv(data_path_val)
    data_train.head()
    y_train = data_train['type'].values.reshape(-1, 1) * 1.0
    y_val = data_vali['type'].values.reshape(-1, 1) * 1.0
    tokenizer = text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(data_train['smiles'])
    x_train = prep_text(data_train['smiles'], tokenizer, maxlen)
    x_val = prep_text(data_vali['smiles'], tokenizer, maxlen)
    print(x_train.shape)
    print(y_train.shape)
    return x_train, y_train, x_val, y_val
