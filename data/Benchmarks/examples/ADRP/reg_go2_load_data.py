def load_data():
    data_path = args['in']
    df = pd.read_csv(data_path, skiprows=1).values.astype('float32')
    df_y = df[:, 0].astype('float32')
    df_x = df[:, 1:PL].astype(np.float32)
    scaler = StandardScaler()
    df_x = scaler.fit_transform(df_x)
    X_train, X_test, Y_train, Y_test = train_test_split(df_x, df_y,
        test_size=0.2, random_state=42)
    print('x_train shape:', X_train.shape)
    print('x_test shape:', X_test.shape)
    return X_train, Y_train, X_test, Y_test
