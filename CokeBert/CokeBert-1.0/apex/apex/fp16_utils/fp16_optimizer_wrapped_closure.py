def wrapped_closure():
    if self.first_closure_call_this_step:
        self.first_closure_call_this_step = False
    else:
        self._master_params_to_model_params()
    temp_loss = closure()
    while self.overflow:
        scale = self.loss_scaler.loss_scale()
        print(
            'OVERFLOW within closure! Skipping step, reducing loss scale to {}'
            .format(self.loss_scaler.loss_scale()))
        temp_loss = closure()
    return temp_loss
