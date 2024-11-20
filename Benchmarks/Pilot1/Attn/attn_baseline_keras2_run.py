def run(params):
    args = candle.ArgumentStruct(**params)
    seed = args.rng_seed
    candle.set_seed(seed)
    ext = attn.extension_from_parameters(params, 'keras')
    candle.verify_path(params['save_path'])
    prefix = '{}{}'.format(params['save_path'], ext)
    logfile = params['logfile'] if params['logfile'] else prefix + '.log'
    root_fname = 'Agg_attn_bin'
    candle.set_up_logger(logfile, attn.logger, params['verbose'])
    attn.logger.info('Params: {}'.format(params))
    X_train, _Y_train, X_val, _Y_val, X_test, _Y_test = attn.load_data(params,
        seed)
    Y_train = _Y_train['AUC']
    Y_test = _Y_test['AUC']
    Y_val = _Y_val['AUC']
    Y_train_neg, Y_train_pos = np.bincount(Y_train)
    Y_test_neg, Y_test_pos = np.bincount(Y_test)
    Y_val_neg, Y_val_pos = np.bincount(Y_val)
    Y_train_total = Y_train_neg + Y_train_pos
    Y_test_total = Y_test_neg + Y_test_pos
    Y_val_total = Y_val_neg + Y_val_pos
    total = Y_train_total + Y_test_total + Y_val_total
    pos = Y_train_pos + Y_test_pos + Y_val_pos
    print("""Examples:
    Total: {}
    Positive: {} ({:.2f}% of total)
"""
        .format(total, pos, 100 * pos / total))
    nb_classes = params['dense'][-1]
    Y_train = to_categorical(Y_train, nb_classes)
    Y_test = to_categorical(Y_test, nb_classes)
    Y_val = to_categorical(Y_val, nb_classes)
    y_integers = np.argmax(Y_train, axis=1)
    class_weights = compute_class_weight(class_weight='balanced', classes=
        np.unique(y_integers), y=y_integers)
    d_class_weights = dict(enumerate(class_weights))
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)
    PS = X_train.shape[1]
    model = build_attention_model(params, PS)
    kerasDefaults = candle.keras_default_config()
    if params['momentum']:
        kerasDefaults['momentum_sgd'] = params['momentum']
    optimizer = candle.build_optimizer(params['optimizer'], params[
        'learning_rate'], kerasDefaults)
    model.compile(loss=params['loss'], optimizer=optimizer, metrics=['acc',
        tf_auc])
    checkpointer = ModelCheckpoint(filepath=params['save_path'] +
        root_fname + '.autosave.model.h5', verbose=1, save_weights_only=
        False, save_best_only=True)
    csv_logger = CSVLogger('{}/{}.training.log'.format(params['save_path'],
        root_fname))
    reduce_lr = ReduceLROnPlateau(monitor='val_tf_auc', factor=0.2,
        patience=40, verbose=1, mode='auto', min_delta=0.0001, cooldown=3,
        min_lr=1e-09)
    early_stop = EarlyStopping(monitor='val_tf_auc', patience=200, verbose=
        1, mode='auto')
    candle_monitor = candle.CandleRemoteMonitor(params=params)
    candle_monitor = candle.CandleRemoteMonitor(params=params)
    timeout_monitor = candle.TerminateOnTimeOut(params['timeout'])
    tensorboard = TensorBoard(log_dir='tb/tb{}'.format(ext))
    history_logger = LoggingCallback(attn.logger.debug)
    callbacks = [candle_monitor, timeout_monitor, csv_logger, history_logger]
    if params['reduce_lr']:
        callbacks.append(reduce_lr)
    if params['use_cp']:
        callbacks.append(checkpointer)
    if params['use_tb']:
        callbacks.append(tensorboard)
    if params['early_stop']:
        callbacks.append(early_stop)
    epochs = params['epochs']
    batch_size = params['batch_size']
    history = model.fit(X_train, Y_train, class_weight=d_class_weights,
        batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(
        X_val, Y_val), callbacks=callbacks)
    if 'loss' in history.history.keys():
        candle.plot_history(params['save_path'] + root_fname, history, 'loss')
    if 'acc' in history.history.keys():
        candle.plot_history(params['save_path'] + root_fname, history, 'acc')
    if 'tf_auc' in history.history.keys():
        candle.plot_history(params['save_path'] + root_fname, history, 'tf_auc'
            )
    score = model.evaluate(X_test, Y_test, verbose=0)
    Y_predict = model.predict(X_test)
    evaluate_model(params, root_fname, nb_classes, Y_test, _Y_test,
        Y_predict, pos, total, score)
    save_and_test_saved_model(params, model, root_fname, X_train, X_test,
        Y_test)
    attn.logger.handlers = []
    return history
