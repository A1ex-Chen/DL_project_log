def set_tensor_by_indices_to_value(tensor, indices, value):
    value_tensor = tf.zeros_like(tensor) + value
    return tf.where(indices, value_tensor, tensor)
