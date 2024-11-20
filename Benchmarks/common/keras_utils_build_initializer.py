def build_initializer(initializer, kerasDefaults, seed=None, constant=0.0):
    """Set the initializer to the appropriate Keras initializer function
    based on the input string and learning rate. Other required values
    are set to the Keras default values

    Parameters
    ----------
    initializer : string
        String to choose the initializer

        Options recognized: 'constant', 'uniform', 'normal',
        'glorot_uniform', 'lecun_uniform', 'he_normal'

        See the Keras documentation for a full description of the options

    kerasDefaults : list
        List of default parameter values to ensure consistency between frameworks

    seed : integer
        Random number seed

    constant : float
        Constant value (for the constant initializer only)

    Return
    ----------
    The appropriate Keras initializer function
    """
    if initializer == 'constant':
        return initializers.Constant(value=constant)
    elif initializer == 'uniform':
        return initializers.RandomUniform(minval=kerasDefaults[
            'minval_uniform'], maxval=kerasDefaults['maxval_uniform'], seed
            =seed)
    elif initializer == 'normal':
        return initializers.RandomNormal(mean=kerasDefaults['mean_normal'],
            stddev=kerasDefaults['stddev_normal'], seed=seed)
    elif initializer == 'glorot_normal':
        return initializers.glorot_normal(seed=seed)
    elif initializer == 'glorot_uniform':
        return initializers.glorot_uniform(seed=seed)
    elif initializer == 'lecun_uniform':
        return initializers.lecun_uniform(seed=seed)
    elif initializer == 'he_normal':
        return initializers.he_normal(seed=seed)
