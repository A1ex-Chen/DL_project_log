def mobilenet_v1():
    input_shape = [96, 96, 3]
    num_classes = 2
    num_filters = 8
    inputs = Input(shape=input_shape)
    x = inputs
    x = Conv2D(num_filters, kernel_size=3, strides=2, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    num_filters = 2 * num_filters
    x = Conv2D(num_filters, kernel_size=1, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=2, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    num_filters = 2 * num_filters
    x = Conv2D(num_filters, kernel_size=1, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size=1, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=2, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    num_filters = 2 * num_filters
    x = Conv2D(num_filters, kernel_size=1, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size=1, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=2, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    num_filters = 2 * num_filters
    x = Conv2D(num_filters, kernel_size=1, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size=1, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size=1, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size=1, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size=1, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size=1, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=2, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    num_filters = 2 * num_filters
    x = Conv2D(num_filters, kernel_size=1, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size=1, strides=1, padding='same',
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=x.shape[1:3])(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model
