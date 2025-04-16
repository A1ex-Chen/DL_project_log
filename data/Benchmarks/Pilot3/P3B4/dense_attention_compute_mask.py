def compute_mask(self, inputs, mask=None):
    self._validate_call_args(inputs=inputs, mask=mask)
    if mask:
        q_mask = mask[0]
        if q_mask is None:
            return None
        return ops.convert_to_tensor_v2(q_mask)
    return None
