def invert_mask(attention_mask: tf.Tensor):
    """Turns 1->0, 0->1, False->True, True-> False"""
    tf.debugging.assert_rank(attention_mask, 2)
    attention_mask = tf.cast(attention_mask, tf.bool)
    ret = tf.math.logical_not(attention_mask)
    return ret
