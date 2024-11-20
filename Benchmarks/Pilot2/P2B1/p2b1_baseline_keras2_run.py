def run(GP):
    if GP['rng_seed']:
        np.random.seed(GP['rng_seed'])
    else:
        np.random.seed(np.random.randint(10000))
    if not os.path.isdir(GP['home_dir']):
        print('Keras home directory not set')
        sys.exit(0)
    sys.path.append(GP['home_dir'])
    args = candle.ArgumentStruct(**GP)
    candle.verify_path(args.save_path)
    prefix = args.save_path
    logfile = args.logfile if args.logfile else prefix + '.log'
    candle.set_up_logger(logfile, logger, False)
    logger.info('Params: {}'.format(GP))
    import p2b1 as hf
    reload(hf)
    from tensorflow.keras import callbacks
    batch_size = GP['batch_size']
    learning_rate = GP['learning_rate']
    kerasDefaults = candle.keras_default_config()
    import helper
    data_files, fields = p2b1.get_list_of_data_files(GP)
    num_samples = 0
    for f in data_files:
        X, nbrs, resnums = helper.get_data_arrays(f)
        num_samples += X.shape[0]
    X, nbrs, resnums = helper.get_data_arrays(data_files[0])
    print('\nData chunk shape: ', X.shape)
    molecular_hidden_layers = GP['molecular_num_hidden']
    if not molecular_hidden_layers:
        X_train = hf.get_data(X, case=GP['case'])
        input_dim = X_train.shape[1]
    else:
        input_dim = X.shape[1] * molecular_hidden_layers[-1]
    print('\nState AE input/output dimension: ', input_dim)
    molecular_nbrs = np.int(GP['molecular_nbrs'])
    num_molecules = X.shape[1]
    num_beads = X.shape[2]
    if GP['nbr_type'] == 'relative':
        num_loc_features = 3
        loc_feat_vect = ['rel_x', 'rel_y', 'rel_z']
    elif GP['nbr_type'] == 'invariant':
        num_loc_features = 2
        loc_feat_vect = ['rel_dist', 'rel_angle']
    else:
        print('Invalid nbr_type!!')
        exit()
    if not GP['type_bool']:
        num_type_features = 0
        type_feat_vect = []
    else:
        num_type_features = 5
        type_feat_vect = list(fields.keys())[3:8]
    num_features = num_loc_features + num_type_features + num_beads
    dim = np.prod([num_beads, num_features, molecular_nbrs + 1])
    bead_kernel_size = num_features
    molecular_input_dim = dim
    mol_kernel_size = num_beads
    feature_vector = loc_feat_vect + type_feat_vect + list(fields.keys())[8:]
    print('\nMolecular AE input/output dimension: ', molecular_input_dim)
    print(
        '\nData Format:\n[Frames (%s), Molecules (%s), Beads (%s), %s (%s)]' %
        (num_samples, num_molecules, num_beads, feature_vector, num_features))
    print('\nDefine the model and compile')
    opt = candle.build_optimizer(GP['optimizer'], learning_rate, kerasDefaults)
    molecular_nonlinearity = GP['molecular_nonlinearity']
    len_molecular_hidden_layers = len(molecular_hidden_layers)
    conv_bool = GP['conv_bool']
    full_conv_bool = GP['full_conv_bool']
    if conv_bool:
        molecular_model, molecular_encoder = AE_models.conv_dense_mol_auto(
            bead_k_size=bead_kernel_size, mol_k_size=mol_kernel_size,
            weights_path=None, input_shape=(1, molecular_input_dim, 1),
            nonlinearity=molecular_nonlinearity, hidden_layers=
            molecular_hidden_layers, l2_reg=GP['l2_reg'], drop=float(GP[
            'dropout']))
    elif full_conv_bool:
        molecular_model, molecular_encoder = AE_models.full_conv_mol_auto(
            bead_k_size=bead_kernel_size, mol_k_size=mol_kernel_size,
            weights_path=None, input_shape=(1, molecular_input_dim, 1),
            nonlinearity=molecular_nonlinearity, hidden_layers=
            molecular_hidden_layers, l2_reg=GP['l2_reg'], drop=float(GP[
            'dropout']))
    else:
        molecular_model, molecular_encoder = AE_models.dense_auto(weights_path
            =None, input_shape=(molecular_input_dim,), nonlinearity=
            molecular_nonlinearity, hidden_layers=molecular_hidden_layers,
            l2_reg=GP['l2_reg'], drop=float(GP['dropout']))
    if GP['loss'] == 'mse':
        loss_func = 'mse'
    elif GP['loss'] == 'custom':
        loss_func = helper.combined_loss
    molecular_model.compile(optimizer=opt, loss=loss_func, metrics=[
        'mean_squared_error', 'mean_absolute_error'])
    print('\nModel Summary: \n')
    molecular_model.summary()
    drop = GP['dropout']
    mb_epochs = GP['epochs']
    initial_lrate = GP['learning_rate']
    epochs_drop = 1 + int(np.floor(mb_epochs / 3))

    def step_decay(epoch):
        global initial_lrate, epochs_drop, drop
        lrate = initial_lrate * np.power(drop, np.floor((1 + epoch) /
            epochs_drop))
        return lrate
    history = callbacks.History()
    history_logger = candle.LoggingCallback(logger.debug)
    candleRemoteMonitor = candle.CandleRemoteMonitor(params=GP)
    timeoutMonitor = candle.TerminateOnTimeOut(TIMEOUT)
    callbacks = [history, history_logger, candleRemoteMonitor, timeoutMonitor]
    if GP['save_path'] is not None:
        save_path = GP['save_path']
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = '.'
    model_json = molecular_model.to_json()
    with open(save_path + '/model.json', 'w') as json_file:
        json_file.write(model_json)
    encoder_json = molecular_encoder.to_json()
    with open(save_path + '/encoder.json', 'w') as json_file:
        json_file.write(encoder_json)
    print('Saved model to disk')
    if GP['train_bool']:
        ct = hf.Candle_Molecular_Train(molecular_model, molecular_encoder,
            data_files, mb_epochs, callbacks, batch_size=batch_size,
            nbr_type=GP['nbr_type'], save_path=GP['save_path'],
            len_molecular_hidden_layers=len_molecular_hidden_layers,
            molecular_nbrs=molecular_nbrs, conv_bool=conv_bool,
            full_conv_bool=full_conv_bool, type_bool=GP['type_bool'],
            sampling_density=GP['sampling_density'])
        frame_loss, frame_mse = ct.train_ac()
    else:
        frame_mse = []
        frame_loss = []
    return frame_loss, frame_mse
