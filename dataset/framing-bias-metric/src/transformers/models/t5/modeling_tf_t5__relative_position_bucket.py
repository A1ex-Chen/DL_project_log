@staticmethod
def _relative_position_bucket(relative_position, bidirectional=True,
    num_buckets=32, max_distance=128):
    """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += tf.dtypes.cast(tf.math.greater(
            relative_position, 0), tf.int32) * num_buckets
        relative_position = tf.math.abs(relative_position)
    else:
        relative_position = -tf.math.minimum(relative_position, 0)
    max_exact = num_buckets // 2
    is_small = tf.math.less(relative_position, max_exact)
    relative_position_if_large = max_exact + tf.dtypes.cast(tf.math.log(tf.
        dtypes.cast(relative_position, tf.float32) / max_exact) / math.log(
        max_distance / max_exact) * (num_buckets - max_exact), tf.int32)
    relative_position_if_large = tf.math.minimum(relative_position_if_large,
        num_buckets - 1)
    relative_buckets += tf.where(is_small, relative_position,
        relative_position_if_large)
    return relative_buckets
