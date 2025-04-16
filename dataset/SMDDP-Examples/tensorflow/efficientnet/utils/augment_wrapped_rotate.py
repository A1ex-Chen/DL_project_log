def wrapped_rotate(image: tf.Tensor, degrees: float, replace: int) ->tf.Tensor:
    """Applies rotation with wrap/unwrap."""
    image = rotate(wrap(image), degrees=degrees)
    return unwrap(image, replace)
