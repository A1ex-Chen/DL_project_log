def load_data(params, seed):
    if params['train_data'].endswith('h5') or params['train_data'].endswith(
        'hdf5'):
        print('processing h5 in file {}'.format(params['train_data']))
        url = params['data_url']
        file_train = params['train_data']
        train_file = candle.get_file(file_train, url + file_train,
            cache_subdir='Pilot1')
        df_x_train_0 = pd.read_hdf(train_file, 'x_train_0').astype(np.float32)
        df_x_train_1 = pd.read_hdf(train_file, 'x_train_1').astype(np.float32)
        X_train = pd.concat([df_x_train_0, df_x_train_1], axis=1, sort=False)
        del df_x_train_0, df_x_train_1
        df_x_test_0 = pd.read_hdf(train_file, 'x_test_0').astype(np.float32)
        df_x_test_1 = pd.read_hdf(train_file, 'x_test_1').astype(np.float32)
        X_test = pd.concat([df_x_test_0, df_x_test_1], axis=1, sort=False)
        del df_x_test_0, df_x_test_1
        df_x_val_0 = pd.read_hdf(train_file, 'x_val_0').astype(np.float32)
        df_x_val_1 = pd.read_hdf(train_file, 'x_val_1').astype(np.float32)
        X_val = pd.concat([df_x_val_0, df_x_val_1], axis=1, sort=False)
        del df_x_val_0, df_x_val_1
        Y_train = pd.read_hdf(train_file, 'y_train')
        Y_test = pd.read_hdf(train_file, 'y_test')
        Y_val = pd.read_hdf(train_file, 'y_val')
    else:
        print('expecting in file file suffix h5')
        sys.exit()
    print('x_train shape:', X_train.shape)
    print('x_test shape:', X_test.shape)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test
    if train_file.endswith('h5') or train_file.endswith('hdf5'):
        print('processing h5 in file {}'.format(train_file))
        df_x_train_0 = pd.read_hdf(train_file, 'x_train_0').astype(np.float32)
        df_x_train_1 = pd.read_hdf(train_file, 'x_train_1').astype(np.float32)
        X_train = pd.concat([df_x_train_0, df_x_train_1], axis=1, sort=False)
        del df_x_train_0, df_x_train_1
        df_x_test_0 = pd.read_hdf(train_file, 'x_test_0').astype(np.float32)
        df_x_test_1 = pd.read_hdf(train_file, 'x_test_1').astype(np.float32)
        X_test = pd.concat([df_x_test_0, df_x_test_1], axis=1, sort=False)
        del df_x_test_0, df_x_test_1
        df_x_val_0 = pd.read_hdf(train_file, 'x_val_0').astype(np.float32)
        df_x_val_1 = pd.read_hdf(train_file, 'x_val_1').astype(np.float32)
        X_val = pd.concat([df_x_val_0, df_x_val_1], axis=1, sort=False)
        del df_x_val_0, df_x_val_1
        Y_train = pd.read_hdf(train_file, 'y_train')
        Y_test = pd.read_hdf(train_file, 'y_test')
        Y_val = pd.read_hdf(train_file, 'y_val')
    else:
        print('expecting in file file suffix h5')
        sys.exit()
    print('x_train shape:', X_train.shape)
    print('x_test shape:', X_test.shape)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test
