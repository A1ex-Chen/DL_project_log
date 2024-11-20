def gated_conv(inputs, kernel_shape, data_num_channels, scope='gated_conv'):
    with tf.variable_scope(scope):
        horiz_inputs, vert_inputs = tf.split(3, 2, inputs)
        p = get_shape(horiz_inputs)[-1]
        p2 = 2 * p
        vert_nxn = conv(vert_inputs, p2, kernel_shape, 'V',
            data_num_channels, scope='vertical_nxn')
        vert_gated_out = _gated_activation_unit(vert_nxn, kernel_shape, 'V',
            data_num_channels, scope='vertical_gated_activation_unit')
        vert_1x1 = conv(vert_nxn, p2, [1, 1], 'V', data_num_channels, scope
            ='vertical_1x1')
        horiz_1xn = conv(horiz_inputs, p2, kernel_shape, 'B',
            data_num_channels, scope='horizontal_1xn')
        horiz_gated_in = vert_1x1 + horiz_1xn
        horiz_gated_out = _gated_activation_unit(horiz_gated_in,
            kernel_shape, 'B', data_num_channels, scope=
            'horizontal_gated_activation_unit')
        horiz_1x1 = conv(horiz_gated_out, p, kernel_shape, 'B',
            data_num_channels, scope='horizontal_1x1')
        horiz_outputs = horiz_1x1 + horiz_inputs
        return tf.concat(3, [horiz_outputs, vert_gated_out])
