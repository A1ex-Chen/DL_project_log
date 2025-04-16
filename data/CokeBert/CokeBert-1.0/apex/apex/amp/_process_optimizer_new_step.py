def new_step(self, closure=None):
    if closure is not None:
        raise RuntimeError(
            'Currently, Amp does not support closure use with optimizers.')
    retval = old_step()
    self._master_params_to_model_params()
    for param in self._amp_stash.all_fp32_from_fp16_params:
        param.grad = None
    return retval
