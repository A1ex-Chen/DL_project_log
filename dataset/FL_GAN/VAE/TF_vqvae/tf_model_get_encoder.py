def get_encoder(latent_dim=128, input_shape=[]):
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(
        encoder_inputs)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    encoder_outputs = layers.Conv2D(latent_dim, kernel_size=1, padding='same')(
        x)
    return keras.Model(encoder_inputs, encoder_outputs, name='encoder')
