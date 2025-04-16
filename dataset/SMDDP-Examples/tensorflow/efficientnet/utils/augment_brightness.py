def brightness(image: tf.Tensor, factor: float) ->tf.Tensor:
    """Equivalent of PIL Brightness."""
    degenerate = tf.zeros_like(image)
    return blend(degenerate, image, factor)
