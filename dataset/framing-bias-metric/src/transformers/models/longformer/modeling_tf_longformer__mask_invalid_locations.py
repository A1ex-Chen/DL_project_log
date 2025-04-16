@staticmethod
def _mask_invalid_locations(input_tensor, window_overlap):
    mask_2d_upper = tf.reverse(tf.linalg.band_part(tf.ones(shape=(
        window_overlap, window_overlap + 1)), -1, 0), axis=[0])
    padding = tf.constant([[0, shape_list(input_tensor)[1] - window_overlap
        ], [0, shape_list(input_tensor)[3] - window_overlap - 1]])
    mask_2d = tf.pad(mask_2d_upper, padding)
    mask_2d = mask_2d + tf.reverse(mask_2d, axis=[0, 1])
    mask_4d = tf.broadcast_to(mask_2d[None, :, None, :], shape_list(
        input_tensor))
    inf_tensor = -float('inf') * tf.ones_like(input_tensor, dtype=tf.dtypes
        .float32)
    input_tensor = tf.where(tf.math.greater(mask_4d, 0), inf_tensor,
        input_tensor)
    return input_tensor
