def run(gParameters):
    X_train, Y_train, X_test, Y_test = bmk.load_data(gParameters)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)
    x_train_len = X_train.shape[1]
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    model = Sequential()
    dense_first = True
    layer_list = list(range(0, len(gParameters['conv']), 3))
    for _, i in enumerate(layer_list):
        filters = gParameters['conv'][i]
        filter_len = gParameters['conv'][i + 1]
        stride = gParameters['conv'][i + 2]
        print(i / 3, filters, filter_len, stride)
        if gParameters['pool']:
            pool_list = gParameters['pool']
            if type(pool_list) != list:
                pool_list = list(pool_list)
        if filters <= 0 or filter_len <= 0 or stride <= 0:
            break
        dense_first = False
        if 'locally_connected' in gParameters:
            model.add(LocallyConnected1D(filters, filter_len, strides=
                stride, padding='valid', input_shape=(x_train_len, 1)))
        elif i == 0:
            model.add(Conv1D(filters=filters, kernel_size=filter_len,
                strides=stride, padding='valid', input_shape=(x_train_len, 1)))
        else:
            model.add(Conv1D(filters=filters, kernel_size=filter_len,
                strides=stride, padding='valid'))
        model.add(Activation(gParameters['activation']))
        if gParameters['pool']:
            model.add(MaxPooling1D(pool_size=pool_list[i // 3]))
    if not dense_first:
        model.add(Flatten())
    for i, layer in enumerate(gParameters['dense']):
        if layer:
            if i == 0 and dense_first:
                model.add(Dense(layer, input_shape=(x_train_len, 1)))
            else:
                model.add(Dense(layer))
            model.add(Activation(gParameters['activation']))
            if gParameters['dropout']:
                model.add(Dropout(gParameters['dropout']))
    if dense_first:
        model.add(Flatten())
    model.add(Dense(gParameters['classes']))
    model.add(Activation(gParameters['out_activation']))
    model.summary()
    model.compile(loss=gParameters['loss'], optimizer=gParameters[
        'optimizer'], metrics=[gParameters['metrics']])
    output_dir = gParameters['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_name = gParameters['model_name']
    path = '{}/{}.autosave.model.h5'.format(output_dir, model_name)
    checkpointer = ModelCheckpoint(filepath=path, verbose=1,
        save_weights_only=False, save_best_only=True)
    csv_logger = CSVLogger('{}/training.log'.format(output_dir))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=
        10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    history = model.fit(X_train, Y_train, batch_size=gParameters[
        'batch_size'], epochs=gParameters['epochs'], verbose=1,
        validation_data=(X_test, Y_test), callbacks=[checkpointer,
        csv_logger, reduce_lr])
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    model_json = model.to_json()
    with open('{}/{}.model.json'.format(output_dir, model_name), 'w'
        ) as json_file:
        json_file.write(model_json)
    model.save_weights('{}/{}.model.h5'.format(output_dir, model_name))
    print('Saved model to disk')
    json_file = open('{}/{}.model.json'.format(output_dir, model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_json = model_from_json(loaded_model_json)
    loaded_model_json.load_weights('{}/{}.model.h5'.format(output_dir,
        model_name))
    print('Loaded json model from disk')
    loaded_model_json.compile(loss=gParameters['loss'], optimizer=
        gParameters['optimizer'], metrics=[gParameters['metrics']])
    score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)
    print('json Test score:', score_json[0])
    print('json Test accuracy:', score_json[1])
    print('json %s: %.2f%%' % (loaded_model_json.metrics_names[1], 
        score_json[1] * 100))
    return history
