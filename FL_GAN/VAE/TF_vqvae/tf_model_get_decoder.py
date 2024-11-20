def get_decoder(latent_dim=128, input_shape=[]):
    output_channel = input_shape[2]
    latent_inputs = keras.Input(shape=get_encoder(latent_dim, input_shape=
        input_shape).output.shape[1:])
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding
        ='same')(latent_inputs)
    x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding
        ='same')(x)
    decoder_outputs = layers.Conv2DTranspose(output_channel, 3, activation=
        'sigmoid', strides=1, padding='same')(x)
    return keras.Model(latent_inputs, decoder_outputs, name='decoder')
