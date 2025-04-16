def _compute_grad_norm(self, fp16_grads_flat, norm_type=2):
    """
        Compute fp16 grad norm for later clipping(fused with update).
        Internal accumulated in fp32.
        Also fused in NaN check. Possibly other reduction needed for grad.

        Args:
            fp16_grads_flat (tensor): fp16 grad flattened
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the current fp16 gradients (viewed as a single vector).
            Returns -1 if the most recently computed fp16 gradients overflowed
        """
    try:
        norm = float(torch.norm(fp16_grads_flat, 2.0, dtype=torch.float32))
    except TypeError as err:
        norm = float(torch.norm(fp16_grads_flat.float(), 2.0))
    if norm == float('inf') or norm == -float('inf') or norm != norm:
        return -1
    else:
        return norm
