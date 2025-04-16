@tf.keras.utils.register_keras_serializable(package='Text')
def hard_swish(features):
    """Computes a hard version of the swish function.

  This operation can be used to reduce computational cost and improve
  quantization for edge devices.

  Args:
    features: A `Tensor` representing preactivation values.

  Returns:
    The activation value.
  """
    features = tf.convert_to_tensor(features)
    return features * tf.nn.relu6(features + tf.constant(3.0)) * (1.0 / 6.0)
