def shear_y(image: tf.Tensor, level: float, replace: int) ->tf.Tensor:
    """Equivalent of PIL Shearing in Y dimension."""
    image = transform(image=wrap(image), transforms=[1.0, 0.0, 0.0, level, 
        1.0, 0.0, 0.0, 0.0])
    return unwrap(image, replace)
