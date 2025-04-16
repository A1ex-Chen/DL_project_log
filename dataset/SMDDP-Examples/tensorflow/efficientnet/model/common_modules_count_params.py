def count_params(model, trainable_only=True):
    """Returns the count of all model parameters, or just trainable ones."""
    if not trainable_only:
        return model.count_params()
    else:
        return int(np.sum([tf.keras.backend.count_params(p) for p in model.
            trainable_weights]))
