def transformer_model(params):
    embed_dim = params['embed_dim']
    ff_dim = params['ff_dim']
    maxlen = params['maxlen']
    num_heads = params['num_heads']
    vocab_size = params['vocab_size']
    transformer_depth = params['transformer_depth']
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    for i in range(transformer_depth):
        x = transformer_block(x)
    dropout = params['dropout']
    dense_layers = params['dense']
    activation = params['activation']
    out_layer = params['out_layer']
    out_act = params['out_activation']
    x = layers.Reshape((1, 32000), input_shape=(250, 128))(x)
    x = layers.Dropout(0.1)(x)
    for dense in dense_layers:
        x = layers.Dense(dense, activation=activation)(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(out_layer, activation=out_act)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model
