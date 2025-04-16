def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = tf.math.equal(input_ids, padding_idx)
    return padding_mask
