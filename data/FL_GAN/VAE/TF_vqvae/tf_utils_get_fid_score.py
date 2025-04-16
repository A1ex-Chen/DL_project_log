def get_fid_score(real_image, gen_image):
    real_images = real_image
    gen_images = gen_image
    if real_image.shape[3] == 1:
        real_images = tf.tile(real_image, [1, 1, 1, 3])
        gen_images = tf.tile(gen_image, [1, 1, 1, 3])
    size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
    resized_real_images = tf.image.resize(real_images, [size, size], method
        =tf.image.ResizeMethod.BILINEAR)
    resized_generated_images = tf.image.resize(gen_images, [size, size],
        method=tf.image.ResizeMethod.BILINEAR)
    resized_real_images = resized_real_images / 127.5 - 1
    resized_generated_images = resized_generated_images / 127.5 - 1
    inception_model = tfhub.KerasLayer(
        'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5')
    fid = tfgan.eval.frechet_inception_distance(resized_real_images,
        resized_generated_images, classifier_fn=inception_model)
    return fid.numpy()
