def solarize(image: tf.Tensor, threshold: int=128) ->tf.Tensor:
    return tf.where(image < threshold, image, 255 - image)
