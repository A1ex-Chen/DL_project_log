def zero_grad(self, set_grads_to_None=False):
    """
        Zero fp32 and fp16 parameter grads.
        """
    for group in self.optimizer.param_groups:
        for p in group['params']:
            if set_grads_to_None:
                p.grad = None
            elif p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
    for fp16_group in self.fp16_groups:
        for param in fp16_group:
            if set_grads_to_None:
                param.grad = None
            elif param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
