def add_noise(image, std=0.05, seed=None):
    make_noisy = tf.greater(tf.random.uniform([], seed=seed), 0.5)
    image = tf.cond(pred=make_noisy, true_fn=lambda : _add_noise(image, std
        ), false_fn=lambda : image)
    return image
