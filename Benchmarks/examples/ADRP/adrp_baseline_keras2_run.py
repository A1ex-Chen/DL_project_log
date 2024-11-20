def run(params):
    args = candle.ArgumentStruct(**params)
    seed = args.rng_seed
    candle.set_seed(seed)
    prefix = '{}'.format(params['save_path'])
    logfile = params['logfile'] if params['logfile'] else prefix + 'TEST.log'
    candle.set_up_logger(logfile, adrp.logger, params['verbose'])
    adrp.logger.info('Params: {}'.format(params))
    keras_defaults = candle.keras_default_config()
    X_train, Y_train, X_test, Y_test, PS, count_array = adrp.load_data(params,
        seed)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)
    print('Y_test:')
    print(Y_test)
    initializer_weights = candle.build_initializer(params['initialization'],
        keras_defaults, seed)
    initializer_bias = candle.build_initializer('constant', keras_defaults, 0.0
        )
    activation = params['activation']
    out_activation = params['out_activation']
    output_dim = 1
    dense_layers = params['dense']
    inputs = Input(shape=(PS,))
    if dense_layers is not None:
        if type(dense_layers) != list:
            dense_layers = list(dense_layers)
        for i, l in enumerate(dense_layers):
            if i == 0:
                x = Dense(l, activation=activation, kernel_initializer=
                    initializer_weights, bias_initializer=initializer_bias)(
                    inputs)
            else:
                x = Dense(l, activation=activation, kernel_initializer=
                    initializer_weights, bias_initializer=initializer_bias)(x)
            if params['dropout']:
                x = Dropout(params['dropout'])(x)
        output = Dense(output_dim, activation=out_activation,
            kernel_initializer=initializer_weights, bias_initializer=
            initializer_bias)(x)
    else:
        output = Dense(output_dim, activation=out_activation,
            kernel_initializer=initializer_weights, bias_initializer=
            initializer_bias)(inputs)
    model = Model(inputs=inputs, outputs=output)
    model.summary()
    kerasDefaults = candle.keras_default_config()
    if params['momentum']:
        kerasDefaults['momentum_sgd'] = params['momentum']
    optimizer = candle.build_optimizer(params['optimizer'], params[
        'learning_rate'], kerasDefaults)
    model.compile(loss=params['loss'], optimizer=optimizer, metrics=['mae', r2]
        )
    checkpointer = ModelCheckpoint(filepath=params['save_path'] +
        'agg_adrp.autosave.model.h5', verbose=1, save_weights_only=False,
        save_best_only=True)
    csv_logger = CSVLogger(params['save_path'] + 'agg_adrp.training.log')
    min_lr = 1e-09
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience
        =params['reduce_patience'], mode='auto', verbose=1, epsilon=0.0001,
        cooldown=3, min_lr=min_lr)
    early_stop = EarlyStopping(monitor='val_loss', patience=params[
        'early_patience'], verbose=1, mode='auto')
    epochs = params['epochs']
    batch_size = params['batch_size']
    timeout_monitor = candle.TerminateOnTimeOut(params['timeout'])
    if params['use_sample_weight']:
        if params['sample_weight_type'] == 'linear':
            train_weight = np.array(Y_train.values.tolist())
            test_weight = np.array(Y_test.values.tolist())
            print('Linear score weighting')
        elif params['sample_weight_type'] == 'quadratic':
            train_weight = np.square(np.array(Y_train.values.tolist()))
            test_weight = np.square(np.array(Y_test.values.tolist()))
            print('Quadratic score weighting')
        elif params['sample_weight_type'] == 'inverse_samples':
            train_score = np.array(Y_train.values.tolist())
            test_score = np.array(Y_test.values.tolist())
            train_bin = train_score.astype(int)
            test_bin = test_score.astype(int)
            train_count = count_array[train_bin].astype(float)
            test_count = count_array[test_bin].astype(float)
            train_weight = 1.0 / (train_count + 1.0)
            test_weight = 1.0 / (test_count + 1.0)
            print('Inverse sample weighting')
            print('Test score, bin, count, weight:')
            print(test_score[:10,])
            print(test_bin[:10,])
            print(test_count[:10,])
        elif params['sample_weight_type'] == 'inverse_samples_sqrt':
            train_score = np.array(Y_train.values.tolist())
            test_score = np.array(Y_test.values.tolist())
            train_bin = train_score.astype(int)
            test_bin = test_score.astype(int)
            train_count = count_array[train_bin].astype(float)
            test_count = count_array[test_bin].astype(float)
            train_weight = 1.0 / np.sqrt(train_count + 1.0)
            test_weight = 1.0 / np.sqrt(test_count + 1.0)
            print('Inverse sqrt sample weighting')
            print('Test score, bin, count, weight:')
            print(test_score[:10,])
            print(test_bin[:10,])
            print(test_count[:10,])
    else:
        train_weight = np.ones(shape=(len(Y_train),))
        test_weight = np.ones(shape=(len(Y_test),))
    print('Test weight:')
    print(test_weight[:10,])
    print('calling model.fit with epochs={}'.format(epochs))
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=
        epochs, verbose=1, sample_weight=train_weight, validation_data=(
        X_test, Y_test, test_weight), callbacks=[checkpointer,
        timeout_monitor, csv_logger, reduce_lr, early_stop])
    print('Reloading saved best model')
    model.load_weights(params['save_path'] + 'agg_adrp.autosave.model.h5')
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(score)
    print(history.history.keys())
    adrp.logger.handlers = []
    return history
