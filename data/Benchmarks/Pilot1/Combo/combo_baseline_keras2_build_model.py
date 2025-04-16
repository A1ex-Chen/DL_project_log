def build_model(loader, args, verbose=False):
    input_models = {}
    dropout_rate = args.dropout
    permanent_dropout = True
    for fea_type, shape in loader.feature_shapes.items():
        box = build_feature_model(input_shape=shape, name=fea_type,
            dense_layers=args.dense_feature_layers, dropout_rate=
            dropout_rate, permanent_dropout=permanent_dropout)
        if verbose:
            box.summary()
        input_models[fea_type] = box
    inputs = []
    encoded_inputs = []
    for fea_name, fea_type in loader.input_features.items():
        shape = loader.feature_shapes[fea_type]
        fea_input = Input(shape, name='input.' + fea_name)
        inputs.append(fea_input)
        input_model = input_models[fea_type]
        encoded = input_model(fea_input)
        encoded_inputs.append(encoded)
    merged = keras.layers.concatenate(encoded_inputs)
    h = merged
    for i, layer in enumerate(args.dense):
        x = h
        h = Dense(layer, activation=args.activation)(h)
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
    output = Dense(1)(h)
    return Model(inputs, output)
