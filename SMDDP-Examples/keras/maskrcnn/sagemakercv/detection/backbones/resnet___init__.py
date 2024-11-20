def __init__(self, sub_type, data_format='channels_last', trainable=True,
    finetune_bn=False, norm_type='batchnorm', *args, **kwargs):
    """
        Our actual ResNet network.  We return the output of c2, c3,c4,c5
        N.B. batch norm is always run with trained parameters, as we use very small
        batches when training the object layers.

        Args:
        sub_type: model type. Authorized Values: (resnet18, resnet34, resnet50, resnet101, resnet152, resnet200)
        data_format: `str` either "channels_first" for
          `[batch, channels, height, width]` or "channels_last for `[batch, height, width, channels]`.
        finetune_bn: `bool` for whether the model is training.

        Returns the ResNet model for a given size and number of output classes.
        """
    model_params = {'resnet18': {'block': ResidualBlock, 'layers': [2, 2, 2,
        2]}, 'resnet34': {'block': ResidualBlock, 'layers': [3, 4, 6, 3]},
        'resnet50': {'block': BottleneckBlock, 'layers': [3, 4, 6, 3]},
        'resnet101': {'block': BottleneckBlock, 'layers': [3, 4, 23, 3]},
        'resnet152': {'block': BottleneckBlock, 'layers': [3, 8, 36, 3]},
        'resnet200': {'block': BottleneckBlock, 'layers': [3, 24, 36, 3]}}
    if sub_type not in model_params:
        raise ValueError('Not a valid sub_type: %s' % sub_type)
    super(Resnet_Model, self).__init__(*args, trainable=trainable, name=
        sub_type, **kwargs)
    self._finetune_bn = finetune_bn
    self.norm_type = norm_type
    self._data_format = data_format
    self._block_layer = model_params[sub_type]['block']
    self._n_layers = model_params[sub_type]['layers']
    self._local_layers = dict()
    if norm_type == 'batchnorm':
        self._local_layers['conv2d'] = Conv2dFixedPadding(filters=64,
            kernel_size=7, strides=2, data_format=self._data_format,
            trainable=False)
        self._local_layers['batchnorm'] = BNReLULayer(relu=True, init_zero=
            False, data_format=self._data_format, trainable=False)
        self._local_layers['maxpool2d'] = tf.keras.layers.MaxPool2D(pool_size
            =3, strides=2, padding='SAME', data_format=self._data_format)
        self._local_layers['block_1'] = BlockGroup(filters=64, strides=1,
            n_blocks=self._n_layers[0], block_layer=self._block_layer,
            data_format=self._data_format, trainable=False, finetune_bn=False)
        self._local_layers['block_2'] = BlockGroup(filters=128, strides=2,
            n_blocks=self._n_layers[1], block_layer=self._block_layer,
            data_format=self._data_format, trainable=self._trainable,
            finetune_bn=self._finetune_bn)
        self._local_layers['block_3'] = BlockGroup(filters=256, strides=2,
            n_blocks=self._n_layers[2], block_layer=self._block_layer,
            data_format=self._data_format, trainable=self._trainable,
            finetune_bn=self._finetune_bn)
        self._local_layers['block_4'] = BlockGroup(filters=512, strides=2,
            n_blocks=self._n_layers[3], block_layer=self._block_layer,
            data_format=self._data_format, trainable=self._trainable,
            finetune_bn=self._finetune_bn)
    elif norm_type == 'groupnorm':
        self._local_layers['conv2d'] = Conv2dFixedPadding(filters=64,
            kernel_size=7, strides=2, data_format=self._data_format,
            trainable=False)
        self._local_layers['groupnorm'] = GNReLULayer(relu=True, init_zero=
            False, data_format=self._data_format, trainable=True)
        self._local_layers['maxpool2d'] = tf.keras.layers.MaxPool2D(pool_size
            =3, strides=2, padding='SAME', data_format=self._data_format)
        self._local_layers['block_1'] = BlockGroup(filters=64, strides=1,
            n_blocks=self._n_layers[0], block_layer=self._block_layer,
            data_format=self._data_format, trainable=False, finetune_bn=
            False, norm_type=norm_type)
        self._local_layers['block_2'] = BlockGroup(filters=128, strides=2,
            n_blocks=self._n_layers[1], block_layer=self._block_layer,
            data_format=self._data_format, trainable=self._trainable,
            finetune_bn=self._finetune_bn, norm_type=norm_type)
        self._local_layers['block_3'] = BlockGroup(filters=256, strides=2,
            n_blocks=self._n_layers[2], block_layer=self._block_layer,
            data_format=self._data_format, trainable=self._trainable,
            finetune_bn=self._finetune_bn, norm_type=norm_type)
        self._local_layers['block_4'] = BlockGroup(filters=512, strides=2,
            n_blocks=self._n_layers[3], block_layer=self._block_layer,
            data_format=self._data_format, trainable=self._trainable,
            finetune_bn=self._finetune_bn, norm_type=norm_type)
    else:
        raise NotImplementedError
