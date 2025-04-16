def get_vqvae(latent_dim=16, num_embeddings=64, data_shape=[]):
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name=
        'vector_quantizer')
    encoder = get_encoder(latent_dim, data_shape)
    decoder = get_decoder(latent_dim, data_shape)
    inputs = keras.Input(shape=data_shape)
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    return keras.Model(inputs, reconstructions, name='vq_vae')
