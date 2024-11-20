def shear_x(image: tf.Tensor, level: float, replace: int) ->tf.Tensor:
    """Equivalent of PIL Shearing in X dimension."""
    image = transform(image=wrap(image), transforms=[1.0, level, 0.0, 0.0, 
        1.0, 0.0, 0.0, 0.0])
    return unwrap(image, replace)
