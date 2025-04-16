def train_autoencoder(x, y, params={}):
    model_type = params.get('model', 'cvae')
    latent_dim = params.get('latent_dim', 10)
    input_dim = x.shape[1]
    num_classes = y.shape[1]
    x_input = keras.Input(shape=(input_dim,))
    c_input = keras.Input(shape=(num_classes,))
    encoder_input = [x_input, c_input] if model_type == 'cvae' else x_input
    h = layers.concatenate(encoder_input) if model_type == 'cvae' else x_input
    h = layers.Dense(100, activation='relu')(h)
    h = layers.Dense(100, activation='relu')(h)
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
    h = layers.Dense(100, activation='relu')(h)
    h = layers.Dense(100, activation='relu')(h)
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
    batch_size = params.get('batch_size', 256)
    epochs = params.get('epochs', 100)
    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    model.fit(inputs, outputs, batch_size=batch_size, epochs=epochs,
        validation_split=0.1)
    return model
