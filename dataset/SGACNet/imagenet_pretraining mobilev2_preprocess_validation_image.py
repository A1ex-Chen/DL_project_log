def preprocess_validation_image(image, label):
    image = tf.image.resize(image, (256, 256)) / 255.0
    image = tf.image.central_crop(image, central_fraction=224 / 256)
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    mean = tf.reshape(mean, [1, 1, 3])
    image = image - mean
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    std = tf.reshape(std, [1, 1, 3])
    image = image / std
    image = tf.transpose(image, perm=[2, 0, 1])
    return image, label
