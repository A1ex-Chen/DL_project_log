def get_pixel_cnn(pixelcnn_input_shape, K):
    num_residual_blocks = 3
    num_pixelcnn_layers = 3
    pixelcnn_inputs = keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
    ohe = tf.one_hot(pixelcnn_inputs, K)
    x = PixelConvLayer(mask_type='A', filters=128, kernel_size=7,
        activation='relu', padding='same')(ohe)
    for _ in range(num_residual_blocks):
        x = ResidualBlock(filters=128)(x)
    for _ in range(num_pixelcnn_layers):
        x = PixelConvLayer(mask_type='B', filters=128, kernel_size=1,
            strides=1, activation='relu', padding='valid')(x)
    out = keras.layers.Conv2D(filters=K, kernel_size=1, strides=1, padding=
        'valid')(x)
    return keras.Model(pixelcnn_inputs, out, name='pixel_cnn')
