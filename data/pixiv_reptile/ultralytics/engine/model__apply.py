def _apply(self, fn) ->'Model':
    """Apply to(), cpu(), cuda(), half(), float() to model tensors that are not parameters or registered buffers."""
    self._check_is_pytorch_model()
    self = super()._apply(fn)
    self.predictor = None
    self.overrides['device'] = self.device
    return self
