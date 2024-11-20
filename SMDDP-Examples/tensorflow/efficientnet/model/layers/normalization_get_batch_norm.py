def get_batch_norm(batch_norm_type: Text) ->tf.keras.layers.BatchNormalization:
    """A helper to create a batch normalization getter.

  Args:
    batch_norm_type: The type of batch normalization layer implementation. `tpu`
     will use `TpuBatchNormalization`.

  Returns:
    An instance of `tf.keras.layers.BatchNormalization`.
  """
    if batch_norm_type == 'tpu':
        return TpuBatchNormalization
    if batch_norm_type == 'syncbn':
        return SyncBatchNormalization
    return tf.keras.layers.BatchNormalization
