def load_state_dict(self, state_dict):
    """
        Loads a state_dict created by an earlier call to state_dict(). 
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``, 
        whose parameters in turn came from ``model``, it is expected that the user 
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.

        Example::

            model = torch.nn.Linear(D_in, D_out).cuda().half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
    self.loss_scaler = state_dict['loss_scaler']
    self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
    self.overflow = state_dict['overflow']
    self.first_closure_call_this_step = state_dict[
        'first_closure_call_this_step']
    self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    for current_group, saved_group in zip(self.fp32_from_fp16_groups,
        state_dict['fp32_from_fp16']):
        for current, saved in zip(current_group, saved_group):
            current.data.copy_(saved.data)
