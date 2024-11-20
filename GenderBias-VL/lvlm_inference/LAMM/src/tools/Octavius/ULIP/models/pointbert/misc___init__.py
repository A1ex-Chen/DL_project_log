def __init__(self, model, bn_lambda, last_epoch=-1, setter=
    set_bn_momentum_default):
    if not isinstance(model, nn.Module):
        raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(
            type(model).__name__))
    self.model = model
    self.setter = setter
    self.lmbd = bn_lambda
    self.step(last_epoch + 1)
    self.last_epoch = last_epoch
