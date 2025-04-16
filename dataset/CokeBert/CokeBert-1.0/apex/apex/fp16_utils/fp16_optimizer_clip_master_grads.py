def clip_master_grads(self, max_norm, norm_type=2):
    """
        Clips fp32 master gradients via ``torch.nn.utils.clip_grad_norm``.

        Args:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the current fp32 gradients (viewed as a single vector).

        .. warning::
            Returns -1 if the most recently computed fp16 gradients overflowed (that is, if ``self.overflow`` is ``True``).
        """
    if not self.overflow:
        fp32_params = []
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                fp32_params.append(param)
        return self.clip_grad_norm(fp32_params, max_norm, norm_type)
    else:
        return -1
