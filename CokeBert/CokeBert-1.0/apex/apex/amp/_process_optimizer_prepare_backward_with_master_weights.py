def prepare_backward_with_master_weights(self):
    stash = self._amp_stash
    if not stash.lazy_init_called:
        self._lazy_init_maybe_master_weights()
        stash.lazy_init_called = True
    for i, param in enumerate(stash.all_fp16_params):
        param.grad = None
    for i, param in enumerate(stash.all_fp32_from_fp32_params):
        stash.all_fp32_from_fp32_grad_stash[i] = param.grad
        param.grad = None
