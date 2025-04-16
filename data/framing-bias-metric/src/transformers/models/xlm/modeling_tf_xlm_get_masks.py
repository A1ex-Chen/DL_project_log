def get_masks(slen, lengths, causal, padding_mask=None, dtype=tf.float32):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    bs = shape_list(lengths)[0]
    if padding_mask is not None:
        mask = padding_mask
    else:
        alen = tf.range(slen)
        mask = tf.math.less(alen, lengths[:, tf.newaxis])
    if causal:
        attn_mask = tf.less_equal(tf.tile(alen[tf.newaxis, tf.newaxis, :],
            (bs, slen, 1)), alen[tf.newaxis, :, tf.newaxis])
    else:
        attn_mask = mask
    tf.debugging.assert_equal(shape_list(mask), [bs, slen])
    assert causal is False or shape_list(attn_mask) == [bs, slen, slen]
    mask = tf.cast(mask, dtype=dtype)
    attn_mask = tf.cast(attn_mask, dtype=dtype)
    return mask, attn_mask
