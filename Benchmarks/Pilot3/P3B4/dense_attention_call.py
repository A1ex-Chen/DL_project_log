def call(self, inputs, mask=None, training=None):
    self._validate_call_args(inputs=inputs, mask=mask)
    q = inputs[0]
    v = inputs[1]
    k = inputs[2] if len(inputs) > 2 else v
    q_mask = mask[0] if mask else None
    v_mask = mask[1] if mask else None
    scores = self._calculate_scores(query=q, key=k)
    if v_mask is not None:
        v_mask = array_ops.expand_dims(v_mask, axis=-2)
    if self.causal:
        scores_shape = array_ops.shape(scores)
        causal_mask_shape = array_ops.concat([array_ops.ones_like(
            scores_shape[:-2]), scores_shape[-2:]], axis=0)
        causal_mask = _lower_triangular_mask(causal_mask_shape)
    else:
        causal_mask = None
    scores_mask = _merge_masks(v_mask, causal_mask)
    result = self._apply_scores(scores=scores, value=v, scores_mask=
        scores_mask, training=training)
    if q_mask is not None:
        q_mask = array_ops.expand_dims(q_mask, axis=-1)
        result *= math_ops.cast(q_mask, dtype=result.dtype)
    return result
