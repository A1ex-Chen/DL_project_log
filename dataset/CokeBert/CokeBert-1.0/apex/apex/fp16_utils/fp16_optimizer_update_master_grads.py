def update_master_grads(self):
    """
        Copy the ``.grad`` attribute from stored references to fp16 parameters to 
        the ``.grad`` attribute of the fp32 master parameters that are directly 
        updated by the optimizer.  :attr:`update_master_grads` only needs to be called if
        ``fp16_optimizer_obj.backward`` was called with ``update_master_grads=False``.
        """
    self.loss_scaler.clear_overflow_state()
    if len(self.all_fp16_params) > 0:
        model_grads = []
        master_grads = []
        for model_param, master_param in zip(self.all_fp16_params, self.
            all_fp32_from_fp16_params):
            if model_param.grad is not None:
                model_grads.append(model_param.grad)
                if master_param.grad is None:
                    master_param.grad = torch.empty_like(master_param)
                master_grads.append(master_param.grad)
        self.loss_scaler.unscale(model_grads, master_grads, self.
            loss_scaler.loss_scale())
    if len(self.all_fp32_from_fp32_params) > 0:
        model_grads = []
        master_grads = []
        for model_param, master_param in zip(self.all_fp32_from_fp32_params,
            self.all_fp32_from_fp32_params):
            if model_param.grad is not None:
                model_grads.append(model_param.grad)
                master_grads.append(master_param.grad)
        self.loss_scaler.unscale(model_grads, master_grads, self.
            loss_scaler.loss_scale())
    self.overflow = self.loss_scaler.update_scale()
