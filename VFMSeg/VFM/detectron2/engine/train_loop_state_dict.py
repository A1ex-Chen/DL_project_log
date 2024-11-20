def state_dict(self):
    ret = super().state_dict()
    ret['grad_scaler'] = self.grad_scaler.state_dict()
    return ret
