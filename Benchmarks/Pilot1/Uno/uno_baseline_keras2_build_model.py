def build_model(loader, args, permanent_dropout=True, silent=False):
    input_models = {}
    dropout_rate = args.dropout
    initializer = 'glorot_normal' if hasattr(args, 'initialization'
        ) is False else args.initialization
    kernel_initializer = candle.build_initializer(initializer, candle.
        keras_default_config(), args.rng_seed)
    for fea_type, shape in loader.feature_shapes.items():
        base_type = fea_type.split('.')[0]
        if base_type in ['cell', 'drug']:
            if (args.dense_cell_feature_layers is not None and base_type ==
                'cell'):
                dense_feature_layers = args.dense_cell_feature_layers
            elif args.dense_drug_feature_layers is not None and base_type == 'drug':
                dense_feature_layers = args.dense_drug_feature_layers
            else:
                dense_feature_layers = args.dense_feature_layers
            box = build_feature_model(input_shape=shape, name=fea_type,
                dense_layers=dense_feature_layers, kernel_initializer=
                kernel_initializer, dropout_rate=dropout_rate,
                permanent_dropout=permanent_dropout)
            if not silent:
                logger.debug('Feature encoding submodel for %s:', fea_type)
                box.summary(print_fn=logger.debug)
            input_models[fea_type] = box
    inputs = []
    encoded_inputs = []
    for fea_name, fea_type in loader.input_features.items():
        shape = loader.feature_shapes[fea_type]
        fea_input = Input(shape, name='input.' + fea_name)
        inputs.append(fea_input)
        if fea_type in input_models:
            input_model = input_models[fea_type]
            encoded = input_model(fea_input)
        else:
            encoded = fea_input
        encoded_inputs.append(encoded)
    merged = keras.layers.concatenate(encoded_inputs)
    h = merged
    for i, layer in enumerate(args.dense):
        x = h
        h = Dense(layer, activation=args.activation, kernel_initializer=
            kernel_initializer)(h)
        if dropout_rate > 0:
            if permanent_dropout:
                h = PermanentDropout(dropout_rate)(h)
            else:
                h = Dropout(dropout_rate)(h)
        if args.residual:
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass
    output = Dense(1, kernel_initializer=kernel_initializer)(h)
    return Model(inputs, output)
