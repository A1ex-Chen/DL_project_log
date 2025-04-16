def preprocess_for_predict(images: tf.Tensor, image_size: int=IMAGE_SIZE,
    num_channels: int=3, dtype: tf.dtypes.DType=tf.float32) ->tf.Tensor:
    images = tf.reshape(images, [image_size, image_size, num_channels])
    if dtype is not None:
        images = tf.image.convert_image_dtype(images, dtype=dtype)
    return images
