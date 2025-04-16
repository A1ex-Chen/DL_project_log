@staticmethod
def _get_global_attn_indices(is_index_global_attn):
    """ compute global attn indices required throughout forward pass """
    num_global_attn_indices = tf.reduce_sum(tf.cast(is_index_global_attn,
        dtype=tf.dtypes.int32), axis=1)
    max_num_global_attn_indices = tf.reduce_max(num_global_attn_indices)
    is_index_global_attn_nonzero = tf.where(is_index_global_attn)
    is_local_index_global_attn = tf.range(max_num_global_attn_indices
        ) < tf.expand_dims(num_global_attn_indices, axis=-1)
    is_local_index_global_attn_nonzero = tf.where(is_local_index_global_attn)
    is_local_index_no_global_attn_nonzero = tf.where(tf.math.logical_not(
        is_local_index_global_attn))
    return (max_num_global_attn_indices, is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero)
