def _compute_new_dynamic_size(image, min_dimension, max_dimension):
    """Compute new dynamic shape for resize_to_range method."""
    image_shape = tf.shape(input=image)
    orig_height = tf.cast(image_shape[0], dtype=tf.float32)
    orig_width = tf.cast(image_shape[1], dtype=tf.float32)
    num_channels = image_shape[2]
    orig_min_dim = tf.minimum(orig_height, orig_width)
    min_dimension = tf.constant(min_dimension, dtype=tf.float32)
    large_scale_factor = min_dimension / orig_min_dim
    large_height = tf.cast(tf.round(orig_height * large_scale_factor),
        dtype=tf.int32)
    large_width = tf.cast(tf.round(orig_width * large_scale_factor), dtype=
        tf.int32)
    large_size = tf.stack([large_height, large_width])
    if max_dimension:
        orig_max_dim = tf.maximum(orig_height, orig_width)
        max_dimension = tf.constant(max_dimension, dtype=tf.float32)
        small_scale_factor = max_dimension / orig_max_dim
        small_height = tf.cast(tf.round(orig_height * small_scale_factor),
            dtype=tf.int32)
        small_width = tf.cast(tf.round(orig_width * small_scale_factor),
            dtype=tf.int32)
        small_size = tf.stack([small_height, small_width])
        new_size = tf.cond(pred=tf.cast(tf.reduce_max(input_tensor=
            large_size), dtype=tf.float32) > max_dimension, true_fn=lambda :
            small_size, false_fn=lambda : large_size)
    else:
        new_size = large_size
    return tf.stack(tf.unstack(new_size) + [num_channels])
