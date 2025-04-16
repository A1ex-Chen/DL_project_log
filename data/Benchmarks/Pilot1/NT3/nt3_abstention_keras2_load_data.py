def load_data(train_path, test_path, gParameters):
    print('Loading data...')
    df_train = pd.read_csv(train_path, header=None).values.astype('float32')
    df_test = pd.read_csv(test_path, header=None).values.astype('float32')
    print('done')
    print('df_train shape:', df_train.shape)
    print('df_test shape:', df_test.shape)
    seqlen = df_train.shape[1]
    df_y_train = df_train[:, 0].astype('int')
    df_y_test = df_test[:, 0].astype('int')
    Y_train = to_categorical(df_y_train, gParameters['classes'])
    Y_test = to_categorical(df_y_test, gParameters['classes'])
    df_x_train = df_train[:, 1:seqlen].astype(np.float32)
    df_x_test = df_test[:, 1:seqlen].astype(np.float32)
    X_train = df_x_train
    X_test = df_x_test
    scaler = MaxAbsScaler()
    mat = np.concatenate((X_train, X_test), axis=0)
    mat = scaler.fit_transform(mat)
    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]
    return X_train, Y_train, X_test, Y_test
