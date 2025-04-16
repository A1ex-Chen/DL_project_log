@property
def output_module_weight(self) ->Tensor:
    return self.net.fc.weight.detach()
