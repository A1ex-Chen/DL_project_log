def load_data(params, seed):
    header_url = params['header_url']
    dh_dict, th_list = load_headers('descriptor_headers.csv',
        'training_headers.csv', header_url)
    offset = 6
    desc_col_idx = [(dh_dict[key] + offset) for key in th_list]
    url = params['data_url']
    file_train = 'ml.' + params['base_name'
        ] + '.Orderable_zinc_db_enaHLL.sorted.4col.dd.parquet'
    train_file = candle.get_file(file_train, url + file_train, cache_subdir
        ='Pilot1')
    print('Loading data...')
    df = pd.read_parquet(train_file)
    print('done')
    df_y = df['reg'].astype('float32')
    df_x = df.iloc[:, desc_col_idx].astype(np.float32)
    bins = np.arange(0, 20)
    histogram, bin_edges = np.histogram(df_y, bins=bins, density=False)
    print('Histogram of samples (bins, counts)')
    print(bin_edges)
    print(histogram)
    scaler = StandardScaler()
    df_x = scaler.fit_transform(df_x)
    X_train, X_test, Y_train, Y_test = train_test_split(df_x, df_y,
        test_size=0.2, random_state=42)
    print('x_train shape:', X_train.shape)
    print('x_test shape:', X_test.shape)
    return X_train, Y_train, X_test, Y_test, X_train.shape[1], histogram
