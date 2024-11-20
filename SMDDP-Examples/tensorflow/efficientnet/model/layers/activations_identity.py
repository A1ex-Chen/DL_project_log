@tf.keras.utils.register_keras_serializable(package='Text')
def identity(features):
    """Computes the identity function.

  Useful for helping in quantization.

  Args:
    features: A `Tensor` representing preactivation values.

  Returns:
    The activation value.
  """
    features = tf.convert_to_tensor(features)
    return tf.identity(features)
