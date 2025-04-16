def build_model(img_rows, img_cols, activation='relu', kernel_initializer=
    'he_normal'):
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(64, 3, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(inputs)
    conv1 = Conv2D(64, 3, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(pool1)
    conv2 = Conv2D(128, 3, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(pool2)
    conv3 = Conv2D(256, 3, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(pool3)
    conv4 = Conv2D(512, 3, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(1024, 3, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(pool4)
    conv5 = Conv2D(1024, 3, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(conv5)
    drop5 = Dropout(0.5)(conv5)
    up6 = Conv2D(512, 2, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(drop5)
        )
    ch, cw = get_crop_shape(drop4, up6)
    crop_drop4 = Cropping2D(cropping=(ch, cw))(drop4)
    merge6 = Concatenate(axis=3)([crop_drop4, up6])
    conv6 = Conv2D(512, 3, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(merge6)
    conv6 = Conv2D(512, 3, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(conv6)
    up7 = Conv2D(256, 2, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(conv6)
        )
    ch, cw = get_crop_shape(conv3, up7)
    crop_conv3 = Cropping2D(cropping=(ch, cw))(conv3)
    merge7 = Concatenate(axis=3)([crop_conv3, up7])
    conv7 = Conv2D(256, 3, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(merge7)
    conv7 = Conv2D(256, 3, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(conv7)
    up8 = Conv2D(128, 2, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(conv7)
        )
    ch, cw = get_crop_shape(conv2, up8)
    crop_conv2 = Cropping2D(cropping=(ch, cw))(conv2)
    merge8 = Concatenate(axis=3)([crop_conv2, up8])
    conv8 = Conv2D(128, 3, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(merge8)
    conv8 = Conv2D(128, 3, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(conv8)
    up9 = Conv2D(64, 2, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(conv8)
        )
    ch, cw = get_crop_shape(conv1, up9)
    crop_conv1 = Cropping2D(cropping=(ch, cw))(conv1)
    merge9 = Concatenate(axis=3)([crop_conv1, up9])
    conv9 = Conv2D(64, 3, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(merge9)
    conv9 = Conv2D(64, 3, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(conv9)
    conv9 = Conv2D(2, 3, activation=activation, padding='same',
        kernel_initializer=kernel_initializer)(conv9)
    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = ZeroPadding2D(padding=(ch[0], cw[0]))(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs, conv10)
    return model
