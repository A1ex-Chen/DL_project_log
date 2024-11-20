def wrap_optimizer(self, optimizer, num_loss=1):
    return OptimWrapper(optimizer, self, num_loss)
