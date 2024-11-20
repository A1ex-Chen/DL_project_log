def translate_x(image: tf.Tensor, pixels: int, replace: int) ->tf.Tensor:
    """Equivalent of PIL Translate in X dimension."""
    image = translate(wrap(image), [-pixels, 0])
    return unwrap(image, replace)
