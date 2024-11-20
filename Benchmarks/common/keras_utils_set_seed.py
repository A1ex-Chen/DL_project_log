def set_seed(seed):
    """Set the random number seed to the desired value

    Parameters
    ----------
    seed : integer
        Random number seed.
    """
    set_seed_defaultUtils(seed)
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if tf.__version__ < '2.0.0':
            tf.compat.v1.set_random_seed(seed)
        else:
            tf.random.set_seed(seed)
