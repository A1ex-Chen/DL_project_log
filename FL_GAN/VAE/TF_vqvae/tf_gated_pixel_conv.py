def conv(inputs, num_outputs, kernel_shape, mask_type, data_num_channels,
    strides=[1, 1], padding='SAME', activation_fn=None, weights_initializer
    =WEIGHT_INITIALIZER, weights_regularizer=None, biases_initializer=tf.
    zeros_initializer, biases_regularizer=None, scope='conv2d'):
    with tf.variable_scope(scope):
        mask_type = mask_type.lower()
        if mask_type == 'v' and kernel_shape == [1, 1]:
            mask_type = None
        num_inputs = get_shape(inputs)[-1]
        kernel_h, kernel_w = kernel_shape
        stride_h, stride_w = strides
        assert kernel_h % 2 == 1 and kernel_w % 2 == 1, 'kernel height and width should be an odd number'
        weights_shape = [kernel_h, kernel_w, num_inputs, num_outputs]
        weights = tf.get_variable('weights', weights_shape, tf.float32,
            weights_initializer, weights_regularizer)
        if mask_type is not None:
            mask = _create_mask(num_inputs, num_outputs, kernel_shape,
                data_num_channels, mask_type)
            weights *= tf.constant(mask, dtype=tf.float32)
            tf.add_to_collection('conv2d_weights_%s' % mask_type, weights)
        outputs = tf.nn.conv2d(inputs, weights, [1, stride_h, stride_w, 1],
            padding=padding, name='outputs')
        tf.add_to_collection('conv2d_outputs', outputs)
        if biases_initializer != None:
            biases = tf.get_variable('biases', [num_outputs], tf.float32,
                biases_initializer, biases_regularizer)
            outputs = tf.nn.bias_add(outputs, biases, name='outputs_plus_b')
        if activation_fn:
            outputs = activation_fn(outputs, name='outputs_with_fn')
        logger.debug('[conv2d_%s] %s : %s %s -> %s %s' % (mask_type, scope,
            inputs.name, inputs.get_shape(), outputs.name, outputs.get_shape())
            )
        return outputs
