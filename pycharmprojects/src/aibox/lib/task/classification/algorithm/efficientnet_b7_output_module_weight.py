@property
def output_module_weight(self) ->Tensor:
    return self.net._fc.weight.detach()
