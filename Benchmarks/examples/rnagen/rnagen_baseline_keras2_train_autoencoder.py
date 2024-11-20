def train_autoencoder(x, y, gParams):
    model_type = gParams['model']
    latent_dim = gParams['latent_dim']
    input_dim = x.shape[1]
    num_classes = y.shape[1]
    x_input = keras.Input(shape=(input_dim,))
    c_input = keras.Input(shape=(num_classes,))
    encoder_input = [x_input, c_input] if model_type == 'cvae' else x_input
    h = layers.concatenate(encoder_input) if model_type == 'cvae' else x_input
    for i in range(len(gParams['encoder_layers'])):
        h = layers.Dense(gParams['encoder_layers'][i], activation=gParams[
            'encoder_activation'])(h)
    if model_type == 'ae':
        encoded = layers.Dense(latent_dim, activation='relu')(h)
    else:
        z_mean = layers.Dense(latent_dim, name='z_mean')(h)
        z_log_var = layers.Dense(latent_dim, name='z_log_var')(h)
        z = Sampling()([z_mean, z_log_var])
        encoded = [z_mean, z_log_var, z]
    latent_input = keras.Input(shape=(latent_dim,))
    decoder_input = [latent_input, c_input
        ] if model_type == 'cvae' else latent_input
    h = layers.concatenate(decoder_input
        ) if model_type == 'cvae' else latent_input
    for i in range(len(gParams['decoder_layers'])):
        h = layers.Dense(gParams['decoder_layers'][i], activation=gParams[
            'decoder_activation'])(h)
    decoded = layers.Dense(input_dim, activation='sigmoid')(h)
    encoder = keras.Model(encoder_input, encoded, name='encoder')
    decoder = keras.Model(decoder_input, decoded, name='decoder')
    if model_type == 'ae':
        model = keras.Model(encoder_input, decoder(encoded))
        metrics = [xent, corr]
        loss = mse
    else:
        model = VAE(encoder, decoder)
        metrics = [xent, mse, corr]
        loss = None
    inputs = [x, y] if model_type == 'cvae' else x
    outputs = x
    batch_size = gParams['batch_size']
    epochs = gParams['epochs']
    model.compile(optimizer=gParams['optimizer'], loss=loss, metrics=metrics)
    model.fit(inputs, outputs, batch_size=batch_size, epochs=epochs,
        validation_split=0.1)
    return model
