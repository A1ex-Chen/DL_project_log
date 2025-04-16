def _apply_func_with_prob(func: Any, image: tf.Tensor, args: Any, prob: float):
    """Apply `func` to image w/ `args` as input with probability `prob`."""
    assert isinstance(args, tuple)
    should_apply_op = tf.cast(tf.floor(tf.random.uniform([], dtype=tf.
        float32) + prob), tf.bool)
    augmented_image = tf.cond(should_apply_op, lambda : func(image, *args),
        lambda : image)
    return augmented_image
