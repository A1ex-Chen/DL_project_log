def get_activation(identifier):
    """Maps a identifier to a Python function, e.g., "relu" => `tf.nn.relu`.

  It checks string first and if it is one of customized activation not in TF,
  the corresponding activation will be returned. For non-customized activation
  names and callable identifiers, always fallback to tf.keras.activations.get.

  Args:
    identifier: String name of the activation function or callable.

  Returns:
    A Python function corresponding to the activation function.
  """
    if isinstance(identifier, six.string_types):
        name_to_fn = {'gelu': gelu, 'simple_swish': simple_swish,
            'hard_swish': hard_swish, 'identity': identity}
        identifier = str(identifier).lower()
        if identifier in name_to_fn:
            return tf.keras.activations.get(name_to_fn[identifier])
    return tf.keras.activations.get(identifier)
