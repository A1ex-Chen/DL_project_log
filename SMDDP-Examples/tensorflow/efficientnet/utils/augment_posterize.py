def posterize(image: tf.Tensor, bits: int) ->tf.Tensor:
    """Equivalent of PIL Posterize."""
    shift = 8 - bits
    return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)
