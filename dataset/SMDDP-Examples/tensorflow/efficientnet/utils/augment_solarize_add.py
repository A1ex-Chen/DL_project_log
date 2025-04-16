def solarize_add(image: tf.Tensor, addition: int=0, threshold: int=128
    ) ->tf.Tensor:
    added_image = tf.cast(image, tf.int64) + addition
    added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
    return tf.where(image < threshold, added_image, image)
