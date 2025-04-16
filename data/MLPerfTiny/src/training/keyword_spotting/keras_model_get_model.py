def get_model(args):
    model_name = args.model_architecture
    label_count = 12
    model_settings = prepare_model_settings(label_count, args)
    if model_name == 'fc4':
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(
            input_shape=(model_settings['spectrogram_length'],
            model_settings['dct_coefficient_count'])), tf.keras.layers.
            Dense(256, activation='relu'), tf.keras.layers.Dropout(0.2), tf
            .keras.layers.BatchNormalization(), tf.keras.layers.Dense(256,
            activation='relu'), tf.keras.layers.Dropout(0.2), tf.keras.
            layers.BatchNormalization(), tf.keras.layers.Dense(256,
            activation='relu'), tf.keras.layers.Dropout(0.2), tf.keras.
            layers.BatchNormalization(), tf.keras.layers.Dense(
            model_settings['label_count'], activation='softmax')])
    elif model_name == 'ds_cnn':
        print('DS CNN model invoked')
        input_shape = [model_settings['spectrogram_length'], model_settings
            ['dct_coefficient_count'], 1]
        filters = 64
        weight_decay = 0.0001
        regularizer = l2(weight_decay)
        final_pool_size = int(input_shape[0] / 2), int(input_shape[1] / 2)
        inputs = Input(shape=input_shape)
        x = Conv2D(filters, (10, 4), strides=(2, 2), padding='same',
            kernel_regularizer=regularizer)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(rate=0.2)(x)
        x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3, 3), padding
            ='same', kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=
            regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3, 3), padding
            ='same', kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=
            regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3, 3), padding
            ='same', kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=
            regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3, 3), padding
            ='same', kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=
            regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(rate=0.4)(x)
        x = AveragePooling2D(pool_size=final_pool_size)(x)
        x = Flatten()(x)
        outputs = Dense(model_settings['label_count'], activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
    elif model_name == 'td_cnn':
        print('TD CNN model invoked')
        input_shape = [model_settings['spectrogram_length'], model_settings
            ['dct_coefficient_count'], 1]
        print(f'Input shape = {input_shape}')
        filters = 64
        weight_decay = 0.0001
        regularizer = l2(weight_decay)
        inputs = Input(shape=input_shape)
        x = Conv2D(filters, (512, 1), strides=(384, 1), padding='valid',
            kernel_regularizer=regularizer)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(rate=0.2)(x)
        x = Reshape((41, 64, 1))(x)
        x = Conv2D(filters, (10, 4), strides=(2, 2), padding='same',
            kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(rate=0.2)(x)
        x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3, 3), padding
            ='same', kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=
            regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3, 3), padding
            ='same', kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=
            regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3, 3), padding
            ='same', kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=
            regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3, 3), padding
            ='same', kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=
            regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(rate=0.4)(x)
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        outputs = Dense(model_settings['label_count'], activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
    else:
        raise ValueError('Model name {:} not supported'.format(model_name))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.
        learning_rate), loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model
