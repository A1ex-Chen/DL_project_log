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
    self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
    self.cur_scale = state_dict['cur_scale']
    self.cur_iter = state_dict['cur_iter']
    if state_dict['dynamic_loss_scale']:
        self.last_overflow_iter = state_dict['last_overflow_iter']
        self.scale_factor = state_dict['scale_factor']
        self.scale_window = state_dict['scale_window']
    self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    for current, saved in zip(self.fp32_groups_flat, state_dict[
        'fp32_groups_flat']):
        current.data.copy_(saved.data)
