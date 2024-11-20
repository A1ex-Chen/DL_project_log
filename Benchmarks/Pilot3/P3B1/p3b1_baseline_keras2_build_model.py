def build_model(gParameters, kerasDefaults, shared_nnet_spec,
    individual_nnet_spec, input_dim, Y_train, Y_test, verbose=False):
    labels_train = []
    labels_test = []
    n_out_nodes = []
    for idx in range(len(Y_train)):
        truth_train = np.array(Y_train[idx], dtype='int32')
        truth_test = np.array(Y_test[idx], dtype='int32')
        mv = int(np.max(truth_train))
        label_train = np.zeros((len(truth_train), mv + 1))
        for i in range(len(truth_train)):
            label_train[i, truth_train[i]] = 1
        label_test = np.zeros((len(truth_test), mv + 1))
        for i in range(len(truth_test)):
            label_test[i, truth_test[i]] = 1
        labels_train.append(label_train)
        labels_test.append(label_test)
        n_out_nodes.append(mv + 1)
    shared_layers = []
    layer = Input(shape=(input_dim,), name='input')
    shared_layers.append(layer)
    for k in range(len(shared_nnet_spec)):
        layer = Dense(shared_nnet_spec[k], activation=gParameters[
            'activation'], name='shared_layer_' + str(k))(shared_layers[-1])
        shared_layers.append(layer)
        if gParameters['dropout'] > 0:
            layer = Dropout(gParameters['dropout'])(shared_layers[-1])
            shared_layers.append(layer)
    indiv_layers_arr = []
    models = []
    trainable_count = 0
    non_trainable_count = 0
    for idx in range(len(individual_nnet_spec)):
        indiv_layers = [shared_layers[-1]]
        for k in range(len(individual_nnet_spec[idx]) + 1):
            if k < len(individual_nnet_spec[idx]):
                layer = Dense(individual_nnet_spec[idx][k], activation=
                    gParameters['activation'], name='indiv_layer_' + str(
                    idx) + '_' + str(k))(indiv_layers[-1])
                indiv_layers.append(layer)
                if gParameters['dropout'] > 0:
                    layer = Dropout(gParameters['dropout'])(indiv_layers[-1])
                    indiv_layers.append(layer)
            else:
                layer = Dense(n_out_nodes[idx], activation=gParameters[
                    'out_activation'], name='out_' + str(idx))(indiv_layers[-1]
                    )
                indiv_layers.append(layer)
        indiv_layers_arr.append(indiv_layers)
        model = Model(inputs=[shared_layers[0]], outputs=[indiv_layers[-1]])
        param_counts = candle.compute_trainable_params(model)
        trainable_count += param_counts['trainable_params']
        non_trainable_count += param_counts['non_trainable_params']
        models.append(model)
    gParameters['trainable_params'] = trainable_count
    gParameters['non_trainable_params'] = non_trainable_count
    gParameters['total_params'] = trainable_count + non_trainable_count
    optimizer = candle.build_optimizer(gParameters['optimizer'],
        gParameters['learning_rate'], kerasDefaults)
    if verbose:
        for k in range(len(models)):
            model = models[k]
            print('Model: ', k)
            model.summary()
    for k in range(len(models)):
        model = models[k]
        model.compile(loss=gParameters['loss'], optimizer=optimizer,
            metrics=[gParameters['metrics']])
    return models, labels_train, labels_test
