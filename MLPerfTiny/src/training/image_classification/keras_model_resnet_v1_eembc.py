def resnet_v1_eembc():
    input_shape = [32, 32, 3]
    num_classes = 10
    num_filters = 16
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters, kernel_size=3, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    y = Conv2D(num_filters, kernel_size=3, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters, kernel_size=3, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(y)
    y = BatchNormalization()(y)
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)
    num_filters = 32
    y = Conv2D(num_filters, kernel_size=3, strides=2, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters, kernel_size=3, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(y)
    y = BatchNormalization()(y)
    x = Conv2D(num_filters, kernel_size=1, strides=2, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)
    num_filters = 64
    y = Conv2D(num_filters, kernel_size=3, strides=2, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters, kernel_size=3, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(y)
    y = BatchNormalization()(y)
    x = Conv2D(num_filters, kernel_size=1, strides=2, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)
    pool_size = int(np.amin(x.shape[1:3]))
    x = AveragePooling2D(pool_size=pool_size)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer=
        'he_normal')(y)
    model = Model(inputs=inputs, outputs=outputs)
    return model
