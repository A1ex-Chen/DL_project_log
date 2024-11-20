def state_dict(self) ->dict:
    """
        Returns the state of the ExponentialMovingAverage as a dict. This method is used by accelerate during
        checkpointing to save the ema state dict.
        """
    return {'decay': self.decay, 'min_decay': self.min_decay,
        'optimization_step': self.optimization_step, 'update_after_step':
        self.update_after_step, 'use_ema_warmup': self.use_ema_warmup,
        'inv_gamma': self.inv_gamma, 'power': self.power, 'shadow_params':
        self.shadow_params}
