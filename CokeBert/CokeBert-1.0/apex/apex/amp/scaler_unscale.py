def unscale(self, model_grads, master_grads, unused_scale,
    models_are_masters=False):
    if self._has_overflow:
        return
    scale = self._loss_scale
    if scale == 1.0 and models_are_masters and not self.dynamic:
        return
    if LossScaler.has_fused_kernel:
        multi_tensor_applier(LossScaler.multi_tensor_scale_cuda, self.
            _overflow_buf, [model_grads, master_grads], 1.0 / scale)
    else:
        self.unscale_python(model_grads, master_grads, scale)
