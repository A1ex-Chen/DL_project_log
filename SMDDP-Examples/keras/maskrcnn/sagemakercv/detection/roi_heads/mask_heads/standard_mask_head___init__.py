def __init__(self, num_classes=91, mrcnn_resolution=28, is_gpu_inference=
    False, name='mask_head', trainable=True, loss_cfg=dict(
    mrcnn_weight_loss_mask=1.0, label_smoothing=0.0), *args, **kwargs):
    """Mask branch for the Mask-RCNN model.
        Args:
        roi_features: A ROI feature tensor of shape
          [batch_size, num_rois, height_l, width_l, num_filters].
        num_classes: an integer for the number of classes.
        mrcnn_resolution: an integer that is the resolution of masks.
        is_gpu_inference: whether to build the model for GPU inference.
        """
    super(StandardMaskHead, self).__init__(*args, name=name, trainable=
        trainable, **kwargs)
    self._num_classes = num_classes
    self._mrcnn_resolution = mrcnn_resolution
    self._is_gpu_inference = is_gpu_inference
    self._conv_stage1 = list()
    kernel_size = 3, 3
    fan_out = 256
    init_stddev = StandardMaskHead._get_stddev_equivalent_to_msra_fill(
        kernel_size, fan_out)
    for conv_id in range(4):
        self._conv_stage1.append(tf.keras.layers.Conv2D(fan_out,
            kernel_size=kernel_size, strides=(1, 1), padding='same',
            dilation_rate=(1, 1), activation=tf.nn.relu, kernel_initializer
            =tf.random_normal_initializer(stddev=init_stddev),
            bias_initializer=tf.keras.initializers.Zeros(), trainable=
            trainable, name='mask-conv-l%d' % conv_id))
    kernel_size = 2, 2
    fan_out = 256
    init_stddev = StandardMaskHead._get_stddev_equivalent_to_msra_fill(
        kernel_size, fan_out)
    self._conv_stage2 = tf.keras.layers.Conv2DTranspose(fan_out,
        kernel_size=kernel_size, strides=(2, 2), padding='valid',
        activation=tf.nn.relu, kernel_initializer=tf.
        random_normal_initializer(stddev=init_stddev), bias_initializer=tf.
        keras.initializers.Zeros(), trainable=trainable, name='conv5-mask')
    kernel_size = 1, 1
    fan_out = self._num_classes
    init_stddev = StandardMaskHead._get_stddev_equivalent_to_msra_fill(
        kernel_size, fan_out)
    self._conv_stage3 = tf.keras.layers.Conv2D(fan_out, kernel_size=
        kernel_size, strides=(1, 1), padding='valid', kernel_initializer=tf
        .random_normal_initializer(stddev=init_stddev), bias_initializer=tf
        .keras.initializers.Zeros(), trainable=trainable, name=
        'mask_fcn_logits')
    self.loss = MaskRCNNLoss(**loss_cfg)
