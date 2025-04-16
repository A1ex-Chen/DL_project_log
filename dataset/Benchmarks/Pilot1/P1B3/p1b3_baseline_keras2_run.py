def run(gParameters):
    """
    Runs the model using the specified set of parameters

    Args:
       gParameters: a python dictionary containing the parameters (e.g. epoch)
       to run the model with.
    """
    if 'dense' in gParameters:
        dval = gParameters['dense']
        if type(dval) != list:
            res = list(dval)
            gParameters['dense'] = res
        print(gParameters['dense'])
    if 'conv' in gParameters:
        flat = gParameters['conv']
        gParameters['conv'] = [flat[i:i + 3] for i in range(0, len(flat), 3)]
        print('Conv input', gParameters['conv'])
    ext = benchmark.extension_from_parameters(gParameters, '.keras')
    logfile = gParameters['logfile'] if gParameters['logfile'
        ] else gParameters['output_dir'] + ext + '.log'
    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter(
        '[%(asctime)s %(process)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(''))
    sh.setLevel(logging.DEBUG if gParameters['verbose'] else logging.INFO)
    benchmark.logger.setLevel(logging.DEBUG)
    benchmark.logger.addHandler(fh)
    benchmark.logger.addHandler(sh)
    benchmark.logger.info('Params: {}'.format(gParameters))
    kerasDefaults = candle.keras_default_config()
    seed = gParameters['rng_seed']
    loader = benchmark.DataLoader(seed=seed, dtype=gParameters['data_type'],
        val_split=gParameters['val_split'], test_cell_split=gParameters[
        'test_cell_split'], cell_features=gParameters['cell_features'],
        drug_features=gParameters['drug_features'], feature_subsample=
        gParameters['feature_subsample'], scaling=gParameters['scaling'],
        scramble=gParameters['scramble'], min_logconc=gParameters[
        'min_logconc'], max_logconc=gParameters['max_logconc'], subsample=
        gParameters['subsample'], category_cutoffs=gParameters[
        'category_cutoffs'])
    initializer_weights = candle.build_initializer(gParameters[
        'initialization'], kerasDefaults, seed)
    initializer_bias = candle.build_initializer('constant', kerasDefaults, 0.0)
    gen_shape = None
    out_dim = 1
    model = Sequential()
    if 'dense' in gParameters:
        for layer in gParameters['dense']:
            if layer:
                model.add(Dense(layer, input_dim=loader.input_dim,
                    kernel_initializer=initializer_weights,
                    bias_initializer=initializer_bias))
                if gParameters['batch_normalization']:
                    model.add(BatchNormalization())
                model.add(Activation(gParameters['activation']))
                if gParameters['dropout']:
                    model.add(Dropout(gParameters['dropout']))
    else:
        gen_shape = 'add_1d'
        layer_list = list(range(0, len(gParameters['conv'])))
        lc_flag = False
        if 'locally_connected' in gParameters:
            lc_flag = True
        for _, i in enumerate(layer_list):
            if i == 0:
                add_conv_layer(model, gParameters['conv'][i], input_dim=
                    loader.input_dim, locally_connected=lc_flag)
            else:
                add_conv_layer(model, gParameters['conv'][i],
                    locally_connected=lc_flag)
            if gParameters['batch_normalization']:
                model.add(BatchNormalization())
            model.add(Activation(gParameters['activation']))
            if gParameters['pool']:
                model.add(MaxPooling1D(pool_size=gParameters['pool']))
        model.add(Flatten())
    model.add(Dense(out_dim))
    optimizer = candle.build_optimizer(gParameters['optimizer'],
        gParameters['learning_rate'], kerasDefaults)
    model.compile(loss=gParameters['loss'], optimizer=optimizer)
    model.summary()
    benchmark.logger.debug('Model: {}'.format(model.to_json()))
    train_gen = benchmark.DataGenerator(loader, batch_size=gParameters[
        'batch_size'], shape=gen_shape, name='train_gen', cell_noise_sigma=
        gParameters['cell_noise_sigma']).flow()
    val_gen = benchmark.DataGenerator(loader, partition='val', batch_size=
        gParameters['batch_size'], shape=gen_shape, name='val_gen').flow()
    val_gen2 = benchmark.DataGenerator(loader, partition='val', batch_size=
        gParameters['batch_size'], shape=gen_shape, name='val_gen2').flow()
    test_gen = benchmark.DataGenerator(loader, partition='test', batch_size
        =gParameters['batch_size'], shape=gen_shape, name='test_gen').flow()
    train_steps = int(loader.n_train / gParameters['batch_size'])
    val_steps = int(loader.n_val / gParameters['batch_size'])
    test_steps = int(loader.n_test / gParameters['batch_size'])
    if 'train_steps' in gParameters:
        train_steps = gParameters['train_steps']
    if 'val_steps' in gParameters:
        val_steps = gParameters['val_steps']
    if 'test_steps' in gParameters:
        test_steps = gParameters['test_steps']
    checkpointer = ModelCheckpoint(filepath=gParameters['output_dir'] +
        '.model' + ext + '.h5', save_best_only=True)
    progbar = MyProgbarLogger(train_steps * gParameters['batch_size'])
    loss_history = MyLossHistory(progbar=progbar, val_gen=val_gen2,
        test_gen=test_gen, val_steps=val_steps, test_steps=test_steps,
        metric=gParameters['loss'], category_cutoffs=gParameters[
        'category_cutoffs'], ext=ext, pre=gParameters['output_dir'])
    np.random.seed(seed)
    candleRemoteMonitor = candle.CandleRemoteMonitor(params=gParameters)
    history = model.fit(train_gen, steps_per_epoch=train_steps, epochs=
        gParameters['epochs'], validation_data=val_gen, validation_steps=
        val_steps, verbose=0, callbacks=[checkpointer, loss_history,
        progbar, candleRemoteMonitor])
    benchmark.logger.removeHandler(fh)
    benchmark.logger.removeHandler(sh)
    return history
