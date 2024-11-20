def scale_gradient(self, module, grad_in, grad_out):
    return tuple(self.loss_scale * g for g in grad_in)
