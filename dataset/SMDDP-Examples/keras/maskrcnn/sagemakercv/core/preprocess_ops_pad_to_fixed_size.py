def pad_to_fixed_size(data, pad_value, output_shape):
    """Pad data to a fixed length at the first dimension.

    Args:
    data: Tensor to be padded to output_shape.
    pad_value: A constant value assigned to the paddings.
    output_shape: The output shape of a 2D tensor.

    Returns:
    The Padded tensor with output_shape [max_num_instances, dimension].
    """
    max_num_instances = output_shape[0]
    dimension = output_shape[1]
    data = tf.reshape(data, [-1, dimension])
    num_instances = tf.shape(input=data)[0]
    pad_length = max_num_instances - num_instances
    paddings = pad_value * tf.ones([pad_length, dimension])
    padded_data = tf.reshape(tf.concat([data, paddings], axis=0), output_shape)
    return padded_data
