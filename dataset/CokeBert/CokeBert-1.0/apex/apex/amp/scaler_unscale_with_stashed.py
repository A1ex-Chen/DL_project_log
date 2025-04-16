def unscale_with_stashed(self, model_grads, stashed_master_grads, master_grads
    ):
    if self._has_overflow:
        return
    scale = self._loss_scale
    if LossScaler.has_fused_kernel:
        if not LossScaler.warned_unscaling_non_fp32_grad and master_grads[0
            ].dtype == torch.float16:
            print(
                "Warning:  unscaling grads that are not FP32. Unscaling non-fp32 grads may indicate an error. When using Amp, you don't need to call .half() on your model."
                )
            LossScaler.warned_unscaling_non_fp32_grad = True
        multi_tensor_applier(LossScaler.multi_tensor_axpby_cuda, self.
            _overflow_buf, [model_grads, stashed_master_grads, master_grads
            ], 1.0 / scale, 1.0, 0)
    else:
        self.unscale_with_stashed_python(model_grads, stashed_master_grads,
            master_grads, scale)
