def inception_logits(images=inception_images, num_splits=1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.compat.v1.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits=
        num_splits)
    logits = tf.map_fn(fn=tfgan.eval.classifier_fn_from_tfhub(
        INCEPTION_TFHUB, INCEPTION_OUTPUT, True), elems=array_ops.stack(
        generated_images_list), parallel_iterations=8, back_prop=False,
        swap_memory=True, name='RunClassifier')
    logits = array_ops.concat(array_ops.unstack(logits), 0)
    return logits