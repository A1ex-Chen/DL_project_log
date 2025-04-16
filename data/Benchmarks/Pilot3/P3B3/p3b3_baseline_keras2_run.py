def run(gParameters):
    fpath = fetch_data(gParameters)
    kerasDefaults = candle.keras_default_config()
    learning_rate = gParameters['learning_rate']
    batch_size = gParameters['batch_size']
    epochs = gParameters['epochs']
    dropout = gParameters['dropout']
    optimizer = gParameters['optimizer']
    wv_len = gParameters['wv_len']
    filter_sizes = gParameters['filter_sizes']
    filter_sets = gParameters['filter_sets']
    num_filters = gParameters['num_filters']
    emb_l2 = gParameters['emb_l2']
    w_l2 = gParameters['w_l2']
    print('Downloaded........')
    train_x = np.load(fpath + '/train_X.npy')
    train_y = np.load(fpath + '/train_Y.npy')
    test_x = np.load(fpath + '/test_X.npy')
    test_y = np.load(fpath + '/test_Y.npy')
    for task in range(len(train_y[0, :])):
        cat = np.unique(train_y[:, task])
        train_y[:, task] = [np.where(cat == x)[0][0] for x in train_y[:, task]]
        test_y[:, task] = [np.where(cat == x)[0][0] for x in test_y[:, task]]
    run_filter_sizes = []
    run_num_filters = []
    for k in range(filter_sets):
        run_filter_sizes.append(filter_sizes + k)
        run_num_filters.append(num_filters)
    ret = run_cnn(gParameters, train_x, train_y, test_x, test_y,
        learning_rate=learning_rate, batch_size=batch_size, epochs=epochs,
        dropout=dropout, optimizer=optimizer, wv_len=wv_len, filter_sizes=
        run_filter_sizes, num_filters=run_num_filters, emb_l2=emb_l2, w_l2=w_l2
        )
    return ret
