def build_attention_model(params, PS):
    assert len(params['dense']) == len(params['activation'])
    assert len(params['dense']) > 3
    DR = params['dropout']
    inputs = Input(shape=(PS,))
    x = Dense(params['dense'][0], activation=params['activation'][0])(inputs)
    x = BatchNormalization()(x)
    a = Dense(params['dense'][1], activation=params['activation'][1])(x)
    a = BatchNormalization()(a)
    b = Dense(params['dense'][2], activation=params['activation'][2])(x)
    x = ke.layers.multiply([a, b])
    for i in range(3, len(params['dense']) - 1):
        x = Dense(params['dense'][i], activation=params['activation'][i])(x)
        x = BatchNormalization()(x)
        x = Dropout(DR)(x)
    outputs = Dense(params['dense'][-1], activation=params['activation'][-1])(x
        )
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model
