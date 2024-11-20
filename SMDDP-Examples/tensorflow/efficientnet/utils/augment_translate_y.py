def translate_y(image: tf.Tensor, pixels: int, replace: int) ->tf.Tensor:
    """Equivalent of PIL Translate in Y dimension."""
    image = translate(wrap(image), [0, -pixels])
    return unwrap(image, replace)
