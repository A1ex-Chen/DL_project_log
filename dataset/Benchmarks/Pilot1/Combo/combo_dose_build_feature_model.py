def build_feature_model(input_shape, name='', dense_layers=[1000, 1000],
    activation='relu', residual=False, dropout_rate=0, permanent_dropout=True):
    x_input = Input(shape=input_shape)
    h = x_input
    for i, layer in enumerate(dense_layers):
        x = h
        h = Dense(layer, activation=activation)(h)
        if dropout_rate > 0:
            if permanent_dropout:
                h = PermanentDropout(dropout_rate)(h)
            else:
                h = Dropout(dropout_rate)(h)
        if residual:
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass
    model = Model(x_input, h, name=name)
    return model
