def __init__(self, conf, height, width, num_channels):
    logger.info('Building gated_pixel_cnn starts')
    self.data = conf.data
    self.height, self.width, self.channel = height, width, num_channels
    self.pixel_depth = 256
    self.q_levels = q_levels = conf.q_levels
    self.inputs = tf.placeholder(tf.float32, [None, height, width,
        num_channels])
    self.target_pixels = tf.placeholder(tf.int64, [None, height, width,
        num_channels])
    logger.info('Building CONV_IN')
    net = conv(self.inputs, conf.gated_conv_num_feature_maps, [7, 7], 'A',
        num_channels, scope='CONV_IN')
    for idx in range(conf.gated_conv_num_layers):
        scope = 'GATED_CONV%d' % idx
        net = gated_conv(net, [3, 3], num_channels, scope=scope)
        logger.info('Building %s' % scope)
    net = tf.nn.relu(conv(net, conf.output_conv_num_feature_maps, [1, 1],
        'B', num_channels, scope='CONV_OUT0'))
    logger.info('Building CONV_OUT0')
    self.logits = tf.nn.relu(conv(net, q_levels * num_channels, [1, 1], 'B',
        num_channels, scope='CONV_OUT1'))
    logger.info('Building CONV_OUT1')
    if num_channels > 1:
        self.logits = tf.reshape(self.logits, [-1, height, width, q_levels,
            num_channels])
        self.logits = tf.transpose(self.logits, perm=[0, 1, 2, 4, 3])
    flattened_logits = tf.reshape(self.logits, [-1, q_levels])
    target_pixels_loss = tf.reshape(self.target_pixels, [-1])
    logger.info('Building loss and optims')
    self.loss = tf.reduce_mean(tf.nn.
        sparse_softmax_cross_entropy_with_logits(flattened_logits,
        target_pixels_loss))
    flattened_output = tf.nn.softmax(flattened_logits)
    self.output = tf.reshape(flattened_output, [-1, height, width,
        num_channels, q_levels])
    optimizer = tf.train.RMSPropOptimizer(conf.learning_rate)
    grads_and_vars = optimizer.compute_gradients(self.loss)
    new_grads_and_vars = [(tf.clip_by_value(gv[0], -conf.grad_clip, conf.
        grad_clip), gv[1]) for gv in grads_and_vars]
    self.optim = optimizer.apply_gradients(new_grads_and_vars)
    logger.info('Building gated_pixel_cnn finished')
