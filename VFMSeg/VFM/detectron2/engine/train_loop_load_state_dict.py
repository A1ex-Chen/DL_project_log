def load_state_dict(self, state_dict):
    super().load_state_dict(state_dict)
    self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
