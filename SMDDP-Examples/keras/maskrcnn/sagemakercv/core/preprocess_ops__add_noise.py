def _add_noise(image, std):
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=std,
        dtype=image.dtype)
    return image * (1.0 + noise)
