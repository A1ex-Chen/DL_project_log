def transform(image: tf.Tensor, transforms) ->tf.Tensor:
    """Prepares input data for `image_ops.transform`."""
    original_ndims = tf.rank(image)
    transforms = tf.convert_to_tensor(transforms, dtype=tf.float32)
    if transforms.shape.rank == 1:
        transforms = transforms[None]
    image = to_4d(image)
    image = image_ops.transform(images=image, transforms=transforms,
        interpolation='nearest')
    return from_4d(image, original_ndims)
