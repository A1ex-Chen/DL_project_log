def get_scalar_from_tensor(t: tf.Tensor) ->int:
    """Utility function to convert a Tensor to a scalar."""
    t = tf.keras.backend.get_value(t)
    if callable(t):
        return t()
    else:
        return t
