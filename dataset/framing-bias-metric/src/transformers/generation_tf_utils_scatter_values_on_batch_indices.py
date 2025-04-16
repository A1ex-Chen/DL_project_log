def scatter_values_on_batch_indices(values, batch_indices):
    shape = shape_list(batch_indices)
    broad_casted_batch_dims = tf.reshape(tf.broadcast_to(tf.expand_dims(tf.
        range(shape[0]), axis=-1), shape), [1, -1])
    pair_indices = tf.transpose(tf.concat([broad_casted_batch_dims, tf.
        reshape(batch_indices, [1, -1])], 0))
    return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), shape)
