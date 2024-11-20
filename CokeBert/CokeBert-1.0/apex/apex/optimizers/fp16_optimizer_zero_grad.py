def zero_grad(self, set_grads_to_None=True):
    """
        Zero FP16 parameter grads.
        """
    for group in self.fp16_groups:
        for p in group:
            if set_grads_to_None:
                p.grad = None
            elif p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
