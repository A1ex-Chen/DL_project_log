def conv_dense_mol_auto(bead_k_size=20, mol_k_size=12, weights_path=None,
    input_shape=(1, 784), hidden_layers=None, nonlinearity='relu', l2_reg=
    0.01, drop=0.5, bias=False):
    input_img = Input(shape=input_shape)
    if type(hidden_layers) != list:
        hidden_layers = list(hidden_layers)
    for i, l in enumerate(hidden_layers):
        if i == 0:
            encoded = Convolution2D(l, bead_k_size, strides=(1, bead_k_size
                ), padding='same', activation=nonlinearity, input_shape=
                input_shape, kernel_regularizer=l2(l2_reg),
                kernel_initializer='glorot_normal', use_bias=bias)(input_img)
            encoded = BatchNormalization()(encoded)
            encoded = Dropout(drop)(encoded)
        elif i == 1:
            encoded = Convolution2D(l, mol_k_size, strides=(1, mol_k_size),
                padding='same', activation=nonlinearity, kernel_regularizer
                =l2(l2_reg), kernel_initializer='glorot_normal', use_bias=bias
                )(encoded)
            encoded = BatchNormalization()(encoded)
            encoded = Dropout(drop)(encoded)
            encoded = Flatten()(encoded)
        elif i == len(hidden_layers) - 1:
            encoded = Dense(l, activation=nonlinearity, kernel_regularizer=
                l2(l2_reg), kernel_initializer='glorot_normal', use_bias=bias)(
                encoded)
            encoded = BatchNormalization()(encoded)
        else:
            encoded = Dense(l, activation=nonlinearity, kernel_regularizer=
                l2(l2_reg), kernel_initializer='glorot_normal', use_bias=bias)(
                encoded)
            encoded = BatchNormalization()(encoded)
            encoded = Dropout(drop)(encoded)
    for i, l in reversed(list(enumerate(hidden_layers))):
        if i < len(hidden_layers) - 1:
            if i == len(hidden_layers) - 2:
                decoded = Dense(l, activation=nonlinearity,
                    kernel_regularizer=l2(l2_reg), kernel_initializer=
                    'glorot_normal', use_bias=bias)(encoded)
                decoded = BatchNormalization()(decoded)
                decoded = Dropout(drop)(decoded)
            else:
                decoded = Dense(l, activation=nonlinearity,
                    kernel_regularizer=l2(l2_reg), kernel_initializer=
                    'glorot_normal', use_bias=bias)(decoded)
                decoded = BatchNormalization()(decoded)
                decoded = Dropout(drop)(decoded)
    decoded = Dense(input_shape[1], activation='tanh', kernel_regularizer=
        l2(l2_reg), kernel_initializer='glorot_normal', use_bias=bias)(decoded)
    model = Model(inputs=input_img, outputs=decoded)
    encoder = Model(inputs=input_img, outputs=encoded)
    if weights_path:
        print('Loading Model')
        model.load_weights(weights_path)
    return model, encoder
