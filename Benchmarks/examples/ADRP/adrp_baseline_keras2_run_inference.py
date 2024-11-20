def run_inference(params):
    if params['saved_model'] is not None:
        model_file = params['saved_model']
    else:
        model_file = adrp.get_model(params)
    print('Loading model from ', model_file)
    if model_file.endswith('.json'):
        base_model_file = model_file.split('.json')
        json_file = open(model_file, 'r')
        loaded_model = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model)
        loaded_model.load_weights(base_model_file[0] + '.h5')
        print('Loaded json model from disk')
    elif model_file.endswith('.yaml'):
        base_model_file = model_file.split('.yaml')
        yaml_file = open(model_file, 'r')
        loaded_model = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model)
        loaded_model.load_weights(base_model_file[0] + '.h5')
        print('Loaded yaml model from disk')
    elif model_file.endswith('.h5'):
        loaded_model = tf.keras.models.load_model(model_file, compile=False)
        print('Loaded h5 model from disk')
    else:
        sys.exit('Model format should be one of json, yaml or h5')
    loaded_model.compile(optimizer=params['optimizer'], loss=params['loss'],
        metrics=['mae', r2])
    seed = params['rng_seed']
    X_train, Y_train, X_test, Y_test, PS, count_array = adrp.load_data(params,
        seed)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)
    score_train = loaded_model.evaluate(X_train, Y_train, verbose=0)
    print('Training set loss:', score_train[0])
    print('Training set mae:', score_train[1])
    score_test = loaded_model.evaluate(X_test, Y_test, verbose=0)
    print('Validation set loss:', score_test[0])
    print('Validation set mae:', score_test[1])
