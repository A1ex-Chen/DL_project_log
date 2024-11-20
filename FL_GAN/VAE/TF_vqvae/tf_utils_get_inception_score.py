@tf.function
def get_inception_score(images):
    image = images[:1000]
    if images.shape[3] == 1:
        image = tf.tile(images[:1000], [1, 1, 1, 3])
    size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
    resized_images = tf.image.resize(image, [size, size], method=tf.image.
        ResizeMethod.BILINEAR)
    inception_model = tfhub.KerasLayer(
        'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5')
    inc_score = tfgan.eval.inception_score(resized_images[:100],
        classifier_fn=inception_model)
    return inc_score
