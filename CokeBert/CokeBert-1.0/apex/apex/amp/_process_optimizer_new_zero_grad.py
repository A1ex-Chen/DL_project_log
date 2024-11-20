def new_zero_grad(self):
    stash = self._amp_stash
    if not stash.lazy_init_called:
        self._lazy_init_maybe_master_weights()
        stash.lazy_init_called = True
    for param in stash.all_fp16_params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()
    for param in stash.all_fp32_from_fp32_params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()
    for param in self._amp_stash.all_fp32_from_fp16_params:
        param.grad = None
