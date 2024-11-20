def unscale_with_stashed_python(self, model_grads, stashed_master_grads,
    master_grads, scale):
    for model, stashed, master in zip(model_grads, stashed_master_grads,
        master_grads):
        if model is None and stashed is None:
            continue
        else:
            if not LossScaler.warned_unscaling_non_fp32_grad:
                if master.dtype != torch.float32:
                    maybe_print(
                        'Attempting to unscale a grad with type {} '.format
                        (master.type()) +
                        "Unscaling non-fp32 grads may indicate an error. When using Amp, you don't need to call .half() on your model."
                        )
                    LossScaler.warned_unscaling_non_fp32_grad = True
            self._has_overflow = axpby_check_overflow_python(model, stashed,
                master, 1.0 / scale, self.dynamic)
            if self._has_overflow and self.dynamic:
                break
