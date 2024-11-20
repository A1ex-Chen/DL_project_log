def any_requires_grad(self, input):
    for tensor in input:
        if tensor.requires_grad:
            return True
    return False
