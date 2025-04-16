def compute_trainable_params(model):
    """Extract number of parameters from the given Keras model

    Parameters
    -----------
    model : Keras model

    Return
    ----------
    python dictionary that contains trainable_params, non_trainable_params and total_params
    """
    if str(type(model)).startswith("<class 'keras."):
        from keras import backend as K
    else:
        import tensorflow.keras.backend as K
    trainable_count = int(np.sum([K.count_params(w) for w in model.
        trainable_weights]))
    non_trainable_count = int(np.sum([K.count_params(w) for w in model.
        non_trainable_weights]))
    return {'trainable_params': trainable_count, 'non_trainable_params':
        non_trainable_count, 'total_params': trainable_count +
        non_trainable_count}
