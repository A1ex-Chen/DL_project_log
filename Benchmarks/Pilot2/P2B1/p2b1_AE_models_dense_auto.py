def dense_auto(weights_path=None, input_shape=(784,), hidden_layers=None,
    nonlinearity='relu', l2_reg=0.01, drop=0.5):
    input_img = Input(shape=input_shape)
    if type(hidden_layers) != list:
        hidden_layers = list(hidden_layers)
    for i, l in enumerate(hidden_layers):
        if i == 0:
            encoded = Dense(l, activation=nonlinearity, kernel_regularizer=
                l2(l2_reg))(input_img)
            encoded = BatchNormalization()(encoded)
            encoded = Dropout(drop)(encoded)
        elif i == len(hidden_layers) - 1:
            encoded = Dense(l, activation='tanh', kernel_regularizer=l2(l2_reg)
                )(encoded)
            encoded = BatchNormalization()(encoded)
        else:
            encoded = Dense(l, activation=nonlinearity, kernel_regularizer=
                l2(l2_reg))(encoded)
            encoded = BatchNormalization()(encoded)
            encoded = Dropout(drop)(encoded)
    for i, l in reversed(list(enumerate(hidden_layers))):
        if i < len(hidden_layers) - 1:
            if i == len(hidden_layers) - 2:
                decoded = Dense(l, activation=nonlinearity,
                    kernel_regularizer=l2(l2_reg))(encoded)
                decoded = BatchNormalization()(decoded)
                decoded = Dropout(drop)(decoded)
            else:
                decoded = Dense(l, activation=nonlinearity,
                    kernel_regularizer=l2(l2_reg))(decoded)
                decoded = BatchNormalization()(decoded)
                decoded = Dropout(drop)(decoded)
    decoded = Dense(input_shape[0], kernel_regularizer=l2(l2_reg))(decoded)
    model = Model(inputs=input_img, outputs=decoded)
    encoder = Model(inputs=input_img, outputs=encoded)
    if weights_path:
        print('Loading Model')
        model.load_weights(weights_path)
    return model, encoder
