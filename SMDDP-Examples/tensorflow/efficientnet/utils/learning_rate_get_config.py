def get_config(self) ->Mapping[str, Any]:
    return {'rescaled_lr': self._rescaled_lr, 'step_boundaries': self.
        _step_boundaries, 'lr_values': self._lr_values, 'warmup_steps':
        self._warmup_steps}
