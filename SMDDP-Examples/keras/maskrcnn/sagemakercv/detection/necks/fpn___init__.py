def __init__(self, min_level=2, max_level=6, filters=256, trainable=True):
    """Generates multiple scale feature pyramid (FPN).

        Args:
        feats_bottom_up: a dictionary of tensor with level as keys and bottom up
          feature tensors as values. They are the features to generate FPN features.
        min_level: the minimum level number to generate FPN features.
        max_level: the maximum level number to generate FPN features.
        filters: the FPN filter size.

        Returns:
        feats: a dictionary of tensor with level as keys and the generated FPN
          features as values.
        """
    super(FPN, self).__init__(name='fpn', trainable=trainable)
    self._local_layers = dict()
    self._min_level = min_level
    self._max_level = max_level
    self._filters = filters
    self._backbone_max_level = 5
    self._upsample_max_level = (self._backbone_max_level if self._max_level >
        self._backbone_max_level else self._max_level)
    self._local_layers['stage1'] = dict()
    for level in range(self._min_level, self._upsample_max_level + 1):
        self._local_layers['stage1'][level] = tf.keras.layers.Conv2D(filters
            =self._filters, kernel_size=(1, 1), padding='same', name='l%d' %
            level, trainable=trainable)
    self._local_layers['stage2'] = dict()
    for level in range(self._min_level, self._upsample_max_level + 1):
        self._local_layers['stage2'][level] = tf.keras.layers.Conv2D(filters
            =self._filters, strides=(1, 1), kernel_size=(3, 3), padding=
            'same', name='post_hoc_d%d' % level, trainable=trainable)
    self._local_layers['stage3_1'] = dict()
    self._local_layers['stage3_2'] = dict()
    if self._max_level == self._upsample_max_level + 1:
        self._local_layers['stage3_1'] = tf.keras.layers.MaxPool2D(pool_size
            =1, strides=2, padding='valid', name='p%d' % self._max_level,
            trainable=trainable)
    else:
        for level in range(self._upsample_max_level + 1, self._max_level + 1):
            self._local_layers['stage3_2'][level] = tf.keras.layers.Conv2D(
                filters=self._filters, strides=(2, 2), kernel_size=(3, 3),
                padding='same', name='p%d' % level, trainable=trainable)
