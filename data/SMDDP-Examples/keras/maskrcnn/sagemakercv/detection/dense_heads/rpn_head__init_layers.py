def _init_layers(self):
    self.rpn_conv = tf.keras.layers.Conv2D(self.feat_channels, kernel_size=
        (3, 3), strides=(1, 1), activation=tf.nn.relu, bias_initializer=tf.
        keras.initializers.Zeros(), kernel_initializer=tf.
        random_normal_initializer(stddev=0.01), padding='same', trainable=
        self.trainable, name='rpn')
    self.conv_cls = tf.keras.layers.Conv2D(len(self.anchor_generator.
        aspect_ratios * self.anchor_generator.num_scales) * self.
        cls_out_channels, kernel_size=(1, 1), strides=(1, 1),
        bias_initializer=tf.keras.initializers.Zeros(), kernel_initializer=
        tf.random_normal_initializer(stddev=0.01), padding='valid',
        trainable=self.trainable, name='rpn-class')
    self.conv_reg = tf.keras.layers.Conv2D(len(self.anchor_generator.
        aspect_ratios * self.anchor_generator.num_scales) * 4, kernel_size=
        (1, 1), strides=(1, 1), bias_initializer=tf.keras.initializers.
        Zeros(), kernel_initializer=tf.random_normal_initializer(stddev=
        0.01), padding='valid', trainable=self.trainable, name='rpn-box')
