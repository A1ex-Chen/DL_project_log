def step(self, closure=None):
    """
        Not supporting closure.
        """
    grads_groups_flat = []
    norm_groups = []
    skip = False
    for i, group in enumerate(self.fp16_groups):
        grads_groups_flat.append(_flatten_dense_tensors([p.grad for p in
            group]))
        norm_groups.append(self._compute_grad_norm(grads_groups_flat[i]))
        if norm_groups[i] == -1:
            skip = True
    if skip:
        self._update_scale(skip)
        return
    self.optimizer.step(grads=[[g] for g in grads_groups_flat],
        output_params=[[p] for p in self.fp16_groups_flat], scale=self.
        cur_scale, grad_norms=norm_groups)
    for i in range(len(norm_groups)):
        updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i],
            self.fp16_groups[i])
        for p, q in zip(self.fp16_groups[i], updated_params):
            p.data = q.data
    self._update_scale(False)
    return
