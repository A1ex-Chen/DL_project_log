def reduce(self):
    if self.module:
        grads = [param.grad.data for param in self.module.parameters() if 
            param.grad is not None]
        flat_dist_call(grads, dist.all_reduce)
    else:
        flat_dist_call(self.grads, dist.all_reduce)
