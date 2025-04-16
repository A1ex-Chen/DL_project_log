@property
def dtype(self) ->tf.dtypes.DType:
    """Converts the config's dtype string to a tf dtype.

    Returns:
      A mapping from string representation of a dtype to the `tf.dtypes.DType`.

    Raises:
      ValueError if the config's dtype is not supported.

    """
    dtype_map = {'float32': tf.float32, 'bfloat16': tf.bfloat16, 'float16':
        tf.float16, 'fp32': tf.float32, 'bf16': tf.bfloat16}
    try:
        return dtype_map[self._dtype]
    except:
        raise ValueError(
            '{} provided key. Invalid DType provided. Supported types: {}'.
            format(self._dtype, dtype_map.keys()))
