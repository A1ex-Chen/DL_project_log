def load_state_dict(self, state_dict: dict) ->None:
    """
        Args:
        Loads the ExponentialMovingAverage state. This method is used by accelerate during checkpointing to save the
        ema state dict.
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
    state_dict = copy.deepcopy(state_dict)
    self.decay = state_dict.get('decay', self.decay)
    if self.decay < 0.0 or self.decay > 1.0:
        raise ValueError('Decay must be between 0 and 1')
    self.min_decay = state_dict.get('min_decay', self.min_decay)
    if not isinstance(self.min_decay, float):
        raise ValueError('Invalid min_decay')
    self.optimization_step = state_dict.get('optimization_step', self.
        optimization_step)
    if not isinstance(self.optimization_step, int):
        raise ValueError('Invalid optimization_step')
    self.update_after_step = state_dict.get('update_after_step', self.
        update_after_step)
    if not isinstance(self.update_after_step, int):
        raise ValueError('Invalid update_after_step')
    self.use_ema_warmup = state_dict.get('use_ema_warmup', self.use_ema_warmup)
    if not isinstance(self.use_ema_warmup, bool):
        raise ValueError('Invalid use_ema_warmup')
    self.inv_gamma = state_dict.get('inv_gamma', self.inv_gamma)
    if not isinstance(self.inv_gamma, (float, int)):
        raise ValueError('Invalid inv_gamma')
    self.power = state_dict.get('power', self.power)
    if not isinstance(self.power, (float, int)):
        raise ValueError('Invalid power')
    shadow_params = state_dict.get('shadow_params', None)
    if shadow_params is not None:
        self.shadow_params = shadow_params
        if not isinstance(self.shadow_params, list):
            raise ValueError('shadow_params must be a list')
        if not all(isinstance(p, torch.Tensor) for p in self.shadow_params):
            raise ValueError('shadow_params must all be Tensors')
