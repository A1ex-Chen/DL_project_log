def _apply_scores(self, scores, value, scores_mask=None, training=None):
    """Applies attention scores to the given value tensor.

        To use this method in your attention layer, follow the steps:

        * Use `query` tensor of shape `[batch_size, Tq]` and `key` tensor of shape
          `[batch_size, Tv]` to calculate the attention `scores`.
        * Pass `scores` and `value` tensors to this method. The method applies
          `scores_mask`, calculates `attention_distribution = softmax(scores)`, then
          returns `matmul(attention_distribution, value).
        * Apply `query_mask` and return the result.

        Args:
          scores: Scores float tensor of shape `[batch_size, Tq, Tv]`.
          value: Value tensor of shape `[batch_size, Tv, dim]`.
          scores_mask: A boolean mask `Tensor` of shape `[batch_size, 1, Tv]` or
            `[batch_size, Tq, Tv]`. If given, scores at positions where
            `scores_mask==False` do not contribute to the result. It must contain
            at least one `True` value in each line along the last dimension.
          training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (no dropout).

        Returns:
          Tensor of shape `[batch_size, Tq, dim]`.
        """
    if scores_mask is not None:
        padding_mask = math_ops.logical_not(scores_mask)
        scores -= 1000000000.0 * math_ops.cast(padding_mask, dtype=K.floatx())
    if training is None:
        training = K.learning_phase()
    weights = nn.softmax(scores)

    def dropped_weights():
        return nn.dropout(weights, rate=self.dropout)
    weights = smart_cond.smart_cond(training, dropped_weights, lambda :
        array_ops.identity(weights))
    return math_ops.matmul(weights, value)
