def gather_based_on_match(self, input_tensor, unmatched_value, ignored_value):
    """Gathers elements from `input_tensor` based on match results.

    For columns that are matched to a row, gathered_tensor[col] is set to
    input_tensor[match_results[col]]. For columns that are unmatched,
    gathered_tensor[col] is set to unmatched_value. Finally, for columns that
    are ignored gathered_tensor[col] is set to ignored_value.

    Note that the input_tensor.shape[1:] must match with unmatched_value.shape
    and ignored_value.shape

    Args:
      input_tensor: Tensor to gather values from.
      unmatched_value: Constant tensor value for unmatched columns.
      ignored_value: Constant tensor value for ignored columns.

    Returns:
      gathered_tensor: A tensor containing values gathered from input_tensor.
        The shape of the gathered tensor is [match_results.shape[0]] +
        input_tensor.shape[1:].
    """
    input_tensor = tf.concat([tf.stack([ignored_value, unmatched_value]),
        input_tensor], axis=0)
    gather_indices = tf.maximum(self.match_results + 2, 0)
    gathered_tensor = tf.gather(input_tensor, gather_indices)
    return gathered_tensor
