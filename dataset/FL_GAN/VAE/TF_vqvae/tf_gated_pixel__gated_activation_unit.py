def _gated_activation_unit(inputs, kernel_shape, mask_type,
    data_num_channels, scope='gated_activation_unit'):
    with tf.variable_scope(scope):
        p2 = get_shape(inputs)[-1]
        bd_out = conv(inputs, p2, kernel_shape, mask_type,
            data_num_channels, scope='blue_diamond')
        bd_out_0, bd_out_1 = tf.split(3, 2, bd_out)
        tanh_out = tf.tanh(bd_out_0)
        sigmoid_out = tf.sigmoid(bd_out_1)
    return tanh_out * sigmoid_out
